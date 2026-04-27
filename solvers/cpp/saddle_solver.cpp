#include <algorithm>
#include <cmath>
#include <cstdint>
#include <omp.h>
#include <vector>

#ifndef EXPORT
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif
#endif

#define MAX_POINTS 10000

struct SaddlePoint {
  float x;
  float y;
  float S;
};

static const int RING_SIZE = 40;
static const int dx_ring[RING_SIZE] = {-5, -4, -3, -2, -1, 0,  1,  2,  3,  4,
                                       5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
                                       5,  4,  3,  2,  1,  0,  -1, -2, -3, -4,
                                       -5, -5, -5, -5, -5, -5, -5, -5, -5, -5};
static const int dy_ring[RING_SIZE] = {-5, -5, -5, -5, -5, -5, -5, -5, -5, -5,
                                       -5, -4, -3, -2, -1, 0,  1,  2,  3,  4,
                                       5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
                                       5,  4,  3,  2,  1,  0,  -1, -2, -3, -4};

// --- Persistent Buffers ---
static std::vector<float> g_blur;
static std::vector<float> g_gx;
static std::vector<float> g_gy;
static std::vector<float> g_mag;
static std::vector<float> g_S;
static std::vector<float> g_sub_s;
static std::vector<float> g_sub_t;
static std::vector<SaddlePoint> g_out_pts;
static int g_capacity_w = 0;
static int g_capacity_h = 0;

extern "C" {

EXPORT void free_saddle_resources_cpp() {
  g_blur.clear();
  g_blur.shrink_to_fit();
  g_gx.clear();
  g_gx.shrink_to_fit();
  g_gy.clear();
  g_gy.shrink_to_fit();
  g_mag.clear();
  g_mag.shrink_to_fit();
  g_S.clear();
  g_S.shrink_to_fit();
  g_sub_s.clear();
  g_sub_s.shrink_to_fit();
  g_sub_t.clear();
  g_sub_t.shrink_to_fit();
  g_out_pts.clear();
  g_out_pts.shrink_to_fit();
  g_capacity_w = 0;
  g_capacity_h = 0;
}

EXPORT int find_saddle_points_cpp(const uint8_t *h_img, SaddlePoint *h_out_pts,
                                  int w, int h, bool filter_t_corners) {
  if (w != g_capacity_w || h != g_capacity_h) {
    free_saddle_resources_cpp();
    size_t pixels = w * h;
    g_blur.resize(pixels);
    g_gx.resize(pixels);
    g_gy.resize(pixels);
    g_mag.resize(pixels);
    g_S.resize(pixels);
    g_sub_s.resize(pixels);
    g_sub_t.resize(pixels);
    g_out_pts.resize(MAX_POINTS);
    g_capacity_w = w;
    g_capacity_h = h;
  }

// 1. Box Blur
#pragma omp parallel for
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      float sum = 0.0f;
      int count = 0;
      for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
          int ny = y + dy;
          int nx = x + dx;
          if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
            sum += h_img[ny * w + nx];
            count++;
          }
        }
      }
      g_blur[y * w + x] = std::round(sum / count);
    }
  }

// 2. Sobel X & Y
#pragma omp parallel for
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      if (x < 1 || y < 1 || x >= w - 1 || y >= h - 1) {
        g_gx[y * w + x] = 0.0f;
        g_gy[y * w + x] = 0.0f;
        continue;
      }
      float p00 = g_blur[(y - 1) * w + x - 1];
      float p01 = g_blur[(y - 1) * w + x];
      float p02 = g_blur[(y - 1) * w + x + 1];
      float p10 = g_blur[y * w + x - 1];
      float p12 = g_blur[y * w + x + 1];
      float p20 = g_blur[(y + 1) * w + x - 1];
      float p21 = g_blur[(y + 1) * w + x];
      float p22 = g_blur[(y + 1) * w + x + 1];

      g_gx[y * w + x] = -p00 + p02 - 2.0f * p10 + 2.0f * p12 - p20 + p22;
      g_gy[y * w + x] = -p00 - 2.0f * p01 - p02 + p20 + 2.0f * p21 + p22;
    }
  }

// 2.5 Calculate Magnitude
#pragma omp parallel for
  for (int i = 0; i < w * h; i++) {
    g_mag[i] = std::sqrt(g_gx[i] * g_gx[i] + g_gy[i] * g_gy[i]);
  }

// 3. Second Order Derivatives & Saddle values
#pragma omp parallel for
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      if (x < 1 || y < 1 || x >= w - 1 || y >= h - 1) {
        g_S[y * w + x] = 0.0f;
        g_sub_s[y * w + x] = 0.0f;
        g_sub_t[y * w + x] = 0.0f;
        continue;
      }

      float g00 = g_gx[(y - 1) * w + x - 1];
      float g02 = g_gx[(y - 1) * w + x + 1];
      float g10 = g_gx[y * w + x - 1];
      float g12 = g_gx[y * w + x + 1];
      float g20 = g_gx[(y + 1) * w + x - 1];
      float g22 = g_gx[(y + 1) * w + x + 1];
      float gxx = -g00 + g02 - 2.0f * g10 + 2.0f * g12 - g20 + g22;

      float g01 = g_gx[(y - 1) * w + x];
      float g21 = g_gx[(y + 1) * w + x];
      float gxy = -g00 - 2.0f * g01 - g02 + g20 + 2.0f * g21 + g22;

      float y00 = g_gy[(y - 1) * w + x - 1];
      float y01 = g_gy[(y - 1) * w + x];
      float y02 = g_gy[(y - 1) * w + x + 1];
      float y20 = g_gy[(y + 1) * w + x - 1];
      float y21 = g_gy[(y + 1) * w + x];
      float y22 = g_gy[(y + 1) * w + x + 1];
      float gyy = -y00 - 2.0f * y01 - y02 + y20 + 2.0f * y21 + y22;

      float S = -gxx * gyy + gxy * gxy;
      g_S[y * w + x] = S;

      float denom = gxx * gyy - gxy * gxy;
      float gx = g_gx[y * w + x];
      float gy = g_gy[y * w + x];

      if (std::abs(denom) > 1e-6f) {
        g_sub_s[y * w + x] = (gy * gxy - gx * gyy) / denom;
        g_sub_t[y * w + x] = (gx * gxy - gy * gxx) / denom;
      } else {
        g_sub_s[y * w + x] = 0.0f;
        g_sub_t[y * w + x] = 0.0f;
      }
    }
  }

  int winsize = 10;
  int out_count = 0;

// 4. Extract Peaks and Filter
#pragma omp parallel for
  for (int y = winsize + 1; y < h - winsize - 1; y++) {
    for (int x = winsize + 1; x < w - winsize - 1; x++) {
      float S = g_S[y * w + x];
      if (S < 10000.0f)
        continue;

      bool is_max = true;
      for (int dy = -5; dy <= 5; dy++) {
        for (int dx = -5; dx <= 5; dx++) {
          if (dx == 0 && dy == 0)
            continue;
          if (g_S[(y + dy) * w + (x + dx)] > S) {
            is_max = false;
            break;
          }
        }
        if (!is_max)
          break;
      }

      if (!is_max)
        continue;

      float px = x + g_sub_s[y * w + x];
      float py = y + g_sub_t[y * w + x];

      if (px <= winsize || py <= winsize || px >= w - winsize - 1.0f ||
          py >= h - winsize - 1.0f)
        continue;

      if (filter_t_corners) {
        float ring_mags[RING_SIZE];
        float ring_intensities[RING_SIZE];
        float sum_mag = 0.0f;
        float sum_intensity = 0.0f;

        // Use rounded subpixel coordinates for the ring center (matches Python
        // np.round)
        int ix = (int)std::rint(px);
        int iy = (int)std::rint(py);

        // Safety clip (matches Python's ixs_safe)
        if (ix < 5)
          ix = 5;
        if (ix > w - 6)
          ix = w - 6;
        if (iy < 5)
          iy = 5;
        if (iy > h - 6)
          iy = h - 6;

        for (int i = 0; i < RING_SIZE; i++) {
          int nx = ix + dx_ring[i];
          int ny = iy + dy_ring[i];
          int idx = ny * w + nx;
          ring_mags[i] = g_mag[idx];
          sum_mag += ring_mags[i];

          ring_intensities[i] = g_blur[idx];
          sum_intensity += ring_intensities[i];
        }

        // 1. Magnitude symmetry score (Rose Plot Symmetry)
        float score = 0.0f;
        if (sum_mag > 1e-6f) {
          float sum_prod = 0.0f;
          for (int i = 0; i < RING_SIZE; i++) {
            sum_prod +=
                ring_mags[i] * ring_mags[(i + RING_SIZE / 2) % RING_SIZE];
          }
          score = sum_prod / (sum_mag * sum_mag);
        }

        // 2. Intensity symmetry (NCC on the ring)
        float intensity_mean = sum_intensity / (float)RING_SIZE;
        float num = 0.0f;
        float den = 0.0f;
        for (int i = 0; i < RING_SIZE; i++) {
          float v1 = ring_intensities[i] - intensity_mean;
          float v2 = ring_intensities[(i + RING_SIZE / 2) % RING_SIZE] -
                     intensity_mean;
          num += v1 * v2;
          den += v1 * v1;
        }
        float ncc = (den > 1e-6f) ? (num / den) : 0.0f;

        // Combined filter: high magnitude symmetry AND high intensity symmetry
        if (score < 0.02f || ncc < 0.2f)
          continue;
      }

      int idx;
#pragma omp critical
      {
        idx = out_count;
        out_count++;
      }

      if (idx < MAX_POINTS) {
        g_out_pts[idx].x = px;
        g_out_pts[idx].y = py;
        g_out_pts[idx].S = S;
      }
    }
  }

  int count = std::min(out_count, MAX_POINTS);

  if (count > 0) {
    for (int i = 0; i < count; i++) {
      h_out_pts[i] = g_out_pts[i];
    }
  }

  return count;
}

} // extern "C"
