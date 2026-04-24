#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#ifndef EXPORT
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif
#endif

#define MAX_POINTS 1000

// --- Output Structure ---
struct SaddlePoint {
    float x;
    float y;
    float S;
};

// --- Persistent Buffers ---
static float* g_d_blur = nullptr;
static float* g_d_gx = nullptr;
static float* g_d_gy = nullptr;
static float* g_d_S = nullptr;
static float* g_d_sub_s = nullptr;
static float* g_d_sub_t = nullptr;
static SaddlePoint* g_d_out_pts = nullptr;
static int* g_d_out_count = nullptr;
static int g_capacity_w = 0;
static int g_capacity_h = 0;

// --- Kernels ---

__global__ void box_blur_kernel(const uint8_t* src, float* dst, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    
    float sum = 0.0f;
    int count = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int ny = y + dy;
            int nx = x + dx;
            if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                sum += src[ny * w + nx];
                count++;
            }
        }
    }
    dst[y * w + x] = sum / count;
}

__global__ void sobel_kernel(const float* src, float* gx, float* gy, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < 1 || y < 1 || x >= w - 1 || y >= h - 1) {
        if (x < w && y < h) {
            gx[y * w + x] = 0.0f;
            gy[y * w + x] = 0.0f;
        }
        return;
    }
    
    float p00 = src[(y - 1) * w + x - 1]; float p01 = src[(y - 1) * w + x]; float p02 = src[(y - 1) * w + x + 1];
    float p10 = src[y * w + x - 1];                                         float p12 = src[y * w + x + 1];
    float p20 = src[(y + 1) * w + x - 1]; float p21 = src[(y + 1) * w + x]; float p22 = src[(y + 1) * w + x + 1];

    gx[y * w + x] = -p00 + p02 - 2.0f * p10 + 2.0f * p12 - p20 + p22;
    gy[y * w + x] = -p00 - 2.0f * p01 - p02 + p20 + 2.0f * p21 + p22;
}

__global__ void sobel_second_and_saddle(const float* gx_img, const float* gy_img, float* S_img, float* sub_s_img, float* sub_t_img, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < 1 || y < 1 || x >= w - 1 || y >= h - 1) {
        if (x < w && y < h) {
            S_img[y * w + x] = 0.0f;
            sub_s_img[y * w + x] = 0.0f;
            sub_t_img[y * w + x] = 0.0f;
        }
        return;
    }

    // GXX is Sobel X of GX
    float g00 = gx_img[(y - 1) * w + x - 1]; float g02 = gx_img[(y - 1) * w + x + 1];
    float g10 = gx_img[y * w + x - 1];       float g12 = gx_img[y * w + x + 1];
    float g20 = gx_img[(y + 1) * w + x - 1]; float g22 = gx_img[(y + 1) * w + x + 1];
    float gxx = -g00 + g02 - 2.0f * g10 + 2.0f * g12 - g20 + g22;

    // GXY is Sobel Y of GX
    float g01 = gx_img[(y - 1) * w + x]; 
    float g21 = gx_img[(y + 1) * w + x];
    float gxy = -g00 - 2.0f * g01 - g02 + g20 + 2.0f * g21 + g22;

    // GYY is Sobel Y of GY
    float y00 = gy_img[(y - 1) * w + x - 1]; float y01 = gy_img[(y - 1) * w + x];   float y02 = gy_img[(y - 1) * w + x + 1];
    float y20 = gy_img[(y + 1) * w + x - 1]; float y21 = gy_img[(y + 1) * w + x];   float y22 = gy_img[(y + 1) * w + x + 1];
    float gyy = -y00 - 2.0f * y01 - y02 + y20 + 2.0f * y21 + y22;

    float S = -gxx * gyy + gxy * gxy;
    S_img[y * w + x] = S;

    float denom = gxx * gyy - gxy * gxy;
    float gx = gx_img[y * w + x];
    float gy = gy_img[y * w + x];
    
    if (denom != 0.0f) {
        sub_s_img[y * w + x] = (gy * gxy - gx * gyy) / denom;
        sub_t_img[y * w + x] = (gx * gxy - gy * gxx) / denom;
    } else {
        sub_s_img[y * w + x] = 0.0f;
        sub_t_img[y * w + x] = 0.0f;
    }
}

__global__ void extract_peaks(const float* S_img, const float* sub_s_img, const float* sub_t_img, const float* blur_img, SaddlePoint* out_pts, int* out_count, bool filter_t_corners, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int winsize = 10;
    
    // Bounds check
    if (x <= winsize || y <= winsize || x >= w - winsize - 1 || y >= h - winsize - 1) return;

    float S = S_img[y * w + x];
    if (S < 10000.0f) return;

    // Fast Non-Max Suppression (11x11 window, radius 5)
    bool is_max = true;
    for (int dy = -5; dy <= 5; dy++) {
        for (int dx = -5; dx <= 5; dx++) {
            if (dx == 0 && dy == 0) continue;
            // Match cv2.dilate(img) == img logic
            // If any neighbor is strictly greater, this is not a peak. 
            // If they are equal, both can be considered peaks, but let's use >.
            if (S_img[(y + dy) * w + (x + dx)] > S) {
                is_max = false;
                break;
            }
        }
        if (!is_max) break;
    }

    if (!is_max) return;

    // Apply subpixel offset and clipping
    float px = x + sub_s_img[y * w + x];
    float py = y + sub_t_img[y * w + x];
    
    // Boundary check again with subpixel coordinates
    if (px <= winsize || py <= winsize || px >= w - winsize - 1.0f || py >= h - winsize - 1.0f) return;

    // T-Corner Filtering (NCC on 11x11 window)
    if (filter_t_corners) {
        int r = 5;
        float patch_mean = 0.0f;
        float patch_rot_mean = 0.0f;
        float num = 0.0f;
        float sum_sq1 = 0.0f;
        float sum_sq2 = 0.0f;
        
        // Compute means
        for (int dy = -r; dy <= r; dy++) {
            for (int dx = -r; dx <= r; dx++) {
                float val1 = blur_img[(y + dy) * w + (x + dx)];
                float val2 = blur_img[(y - dy) * w + (x - dx)];
                patch_mean += val1;
                patch_rot_mean += val2;
            }
        }
        int N = (2 * r + 1) * (2 * r + 1);
        patch_mean /= N;
        patch_rot_mean /= N;

        // Compute NCC
        for (int dy = -r; dy <= r; dy++) {
            for (int dx = -r; dx <= r; dx++) {
                float v1 = blur_img[(y + dy) * w + (x + dx)] - patch_mean;
                float v2 = blur_img[(y - dy) * w + (x - dx)] - patch_rot_mean;
                num += v1 * v2;
                sum_sq1 += v1 * v1;
                sum_sq2 += v2 * v2;
            }
        }
        float den = sqrtf(sum_sq1 * sum_sq2);
        float ncc = (den > 0.0f) ? (num / den) : 0.0f;
        
        // Internal X-corners have NCC > 0 (symmetric). T-corners have NCC < 0 (anti-symmetric).
        if (ncc <= 0.25f) return;
    }

    // Add to global point list safely
    int idx = atomicAdd(out_count, 1);
    if (idx < MAX_POINTS) {
        out_pts[idx].x = px;
        out_pts[idx].y = py;
        out_pts[idx].S = S;
    }
}

// --- C Exports ---
extern "C" {

EXPORT void free_saddle_resources() {
    if (g_d_blur) { cudaFree(g_d_blur); g_d_blur = nullptr; }
    if (g_d_gx) { cudaFree(g_d_gx); g_d_gx = nullptr; }
    if (g_d_gy) { cudaFree(g_d_gy); g_d_gy = nullptr; }
    if (g_d_S) { cudaFree(g_d_S); g_d_S = nullptr; }
    if (g_d_sub_s) { cudaFree(g_d_sub_s); g_d_sub_s = nullptr; }
    if (g_d_sub_t) { cudaFree(g_d_sub_t); g_d_sub_t = nullptr; }
    if (g_d_out_pts) { cudaFree(g_d_out_pts); g_d_out_pts = nullptr; }
    if (g_d_out_count) { cudaFree(g_d_out_count); g_d_out_count = nullptr; }
    g_capacity_w = 0;
    g_capacity_h = 0;
}

EXPORT int find_saddle_points_cuda(const uint8_t* h_img, SaddlePoint* h_out_pts, int w, int h, bool filter_t_corners) {
    if (w != g_capacity_w || h != g_capacity_h) {
        free_saddle_resources();
        
        size_t pixels = w * h;
        cudaMalloc(&g_d_blur, pixels * sizeof(float));
        cudaMalloc(&g_d_gx, pixels * sizeof(float));
        cudaMalloc(&g_d_gy, pixels * sizeof(float));
        cudaMalloc(&g_d_S, pixels * sizeof(float));
        cudaMalloc(&g_d_sub_s, pixels * sizeof(float));
        cudaMalloc(&g_d_sub_t, pixels * sizeof(float));
        
        cudaMalloc(&g_d_out_pts, MAX_POINTS * sizeof(SaddlePoint));
        cudaMalloc(&g_d_out_count, sizeof(int));
        
        g_capacity_w = w;
        g_capacity_h = h;
    }

    // Reset output counter
    int zero = 0;
    cudaMemcpy(g_d_out_count, &zero, sizeof(int), cudaMemcpyHostToDevice);

    // Copy input image directly to GPU. We use a temporary device pointer.
    uint8_t* d_img;
    cudaMalloc(&d_img, w * h * sizeof(uint8_t));
    cudaMemcpy(d_img, h_img, w * h * sizeof(uint8_t), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);

    // 1. Box Blur
    box_blur_kernel<<<grid, block>>>(d_img, g_d_blur, w, h);
    
    // 2. Sobel X & Y
    sobel_kernel<<<grid, block>>>(g_d_blur, g_d_gx, g_d_gy, w, h);
    
    // 3. Second Order Derivatives & Saddle values
    sobel_second_and_saddle<<<grid, block>>>(g_d_gx, g_d_gy, g_d_S, g_d_sub_s, g_d_sub_t, w, h);
    
    // 4. Extract Peaks and Filter
    extract_peaks<<<grid, block>>>(g_d_S, g_d_sub_s, g_d_sub_t, g_d_blur, g_d_out_pts, g_d_out_count, filter_t_corners, w, h);

    cudaDeviceSynchronize();
    cudaFree(d_img);

    // Fetch result
    int count = 0;
    cudaMemcpy(&count, g_d_out_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (count > MAX_POINTS) count = MAX_POINTS;

    if (count > 0) {
        cudaMemcpy(h_out_pts, g_d_out_pts, count * sizeof(SaddlePoint), cudaMemcpyDeviceToHost);
    }

    return count;
}

} // extern "C"
