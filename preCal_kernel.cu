#include "preCal_kernel.h"

#include <opencv2/core/cuda_devptrs.hpp>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include "device_common.h"

#define BLOCK_W 8
#define BLOCK_H 8
#define BLOCK_SIZE1 BLOCK_W*BLOCK_H

using namespace cv;

__global__ void variance_kernel(float* variance, const gpu::PtrStepSz<float> imgptr, const int2 dim);

float getTVperNN(const gpu::PtrStepSz<float> &img, const int &w, const int &h) {
  // allocate
  const int area = w * h;
  thrust::device_vector<float> variance(area, 0.0);
  
  // kernel parameter for TV
  dim3 bDim(BLOCK_W, BLOCK_H);
  dim3 gDim((w - 1)/BLOCK_W + 1, (h - 1)/BLOCK_H + 1);
  variance_kernel<<<gDim,bDim>>>(thrust::raw_pointer_cast(variance.data()), img, make_int2(w, h));
  float TV = thrust::reduce(variance.begin(), variance.end());
  return TV;
}

__global__
void variance_kernel(float* variance, const gpu::PtrStepSz<float> imgptr, const int2 dim) {
  
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  const int tid = tidy*BLOCK_W + tidx;
  const int x = blockIdx.x*BLOCK_W + tidx;
  const int y = blockIdx.y*BLOCK_H + tidy;

  // shared memory
  const int wW = BLOCK_W + 2;
  const int wH = BLOCK_H + 2;
  const int wSize = wW*wH;
  __shared__ float window[wSize];
  
  // move data to shared
  int wXstart = blockIdx.x*BLOCK_W - 1;
  int wYstart = blockIdx.y*BLOCK_H - 1;
  for (int i = tid; i < wSize; i += BLOCK_SIZE1) {
    int wX = (i % wW) + wXstart;
    int wY = (i / wH) + wYstart;
    if (wX < 0 || wX >= dim.x || wY < 0 || wY >= dim.y)
      window[i] = 2;
    else
      window[i] = imgptr(wY, wX);
  }
  __syncthreads();
  
  // out of range
  if (x >= dim.x || y >= dim.y)
    return;
  
  // find max
  float max = 0;
  float value = imgptr(y, x);
  int windowIdx = tidy*wW + tidx;
  for (int idy = tidy; idy < tidy + 3; idy++) {
    for (int idx = tidx; idx < tidx + 3; idx++) {
      float tmp = window[windowIdx++];
      if (tmp != 2) {
        float diff = abs(value - tmp);
        if (diff > max)
          max = diff;
      }
    }
    windowIdx += (BLOCK_W - 1);
  }
  variance[y*dim.x + x] = max;
}
