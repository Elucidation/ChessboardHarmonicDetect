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
    // Match OpenCV uint8 blur rounding
    dst[y * w + x] = roundf(sum / count);
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

__global__ void extract_peaks(const float* S_img, const float* sub_s_img, const float* sub_t_img, const float* gx_img, const float* gy_img, const float* blur_img, SaddlePoint* out_pts, int* out_count, bool filter_t_corners, int w, int h) {
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
    if (px <= winsize || py <= winsize || px >= w - (float)winsize - 1.0f || py >= h - (float)winsize - 1.0f) return;

    if (filter_t_corners) {
        // Ring-based symmetry filtering (Rose Plot and Intensity NCC)
        int dxs[40] = {-5,-4,-3,-2,-1, 0, 1, 2, 3, 4, 5,  5, 5, 5, 5, 5, 5, 5, 5, 5,  5, 4, 3, 2, 1, 0,-1,-2,-3,-4,-5, -5,-5,-5,-5,-5,-5,-5,-5,-5};
        int dys[40] = {-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5, -4,-3,-2,-1, 0, 1, 2, 3, 4,  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,  4, 3, 2, 1, 0,-1,-2,-3,-4};

        float ring_mags[40];
        float ring_vals[40];
        float mag_sum = 0.0f;
        float val_sum = 0.0f;

        // Use rounded subpixel coordinates for the ring center (matches Python np.round)
        int ix = (int)roundf(px);
        int iy = (int)roundf(py);

        // Safety clip (matches Python's ixs_safe)
        if (ix < 5) ix = 5;
        if (ix > w - 6) ix = w - 6;
        if (iy < 5) iy = 5;
        if (iy > h - 6) iy = h - 6;

        for (int i = 0; i < 40; i++) {
            int nx = ix + dxs[i];
            int ny = iy + dys[i];
            float gx = gx_img[ny * w + nx];
            float gy = gy_img[ny * w + nx];
            float mag = sqrtf(gx * gx + gy * gy);
            float val = blur_img[ny * w + nx];
            
            ring_mags[i] = mag;
            ring_vals[i] = val;
            mag_sum += mag;
            val_sum += val;
        }

        // 1. Rose Plot Magnitude Symmetry Score
        float mag_score = 0.0f;
        float mag_norm_denom = mag_sum + 1e-6f;
        for (int i = 0; i < 40; i++) {
            float m1 = ring_mags[i] / mag_norm_denom;
            float m2 = ring_mags[(i + 20) % 40] / mag_norm_denom;
            mag_score += m1 * m2;
        }

        // 2. Intensity NCC Symmetry Score
        float val_mean = val_sum / 40.0f;
        float ncc_num = 0.0f;
        float ncc_den = 0.0f;
        for (int i = 0; i < 40; i++) {
            float v1 = ring_vals[i] - val_mean;
            float v2 = ring_vals[(i + 20) % 40] - val_mean;
            ncc_num += v1 * v2;
            ncc_den += v1 * v1;
        }
        float ncc_score = (ncc_den > 0.0f) ? (ncc_num / ncc_den) : 0.0f;

        // Thresholds matching Python (scores >= 0.02, ncc_scores >= 0.2)
        if (mag_score < 0.02f || ncc_score < 0.2f) return;
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
    extract_peaks<<<grid, block>>>(g_d_S, g_d_sub_s, g_d_sub_t, g_d_gx, g_d_gy, g_d_blur, g_d_out_pts, g_d_out_count, filter_t_corners, w, h);

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
