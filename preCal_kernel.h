#ifndef PRECAL_KERNEL_H
#define PRECAL_KERNEL_H

#include <opencv2/core/cuda_devptrs.hpp>
#include <thrust/device_vector.h>

float getTVperNN(const cv::gpu::PtrStepSz<float> &img, const int &w, const int &h);
//void bindImgtoTex(const cv::gpu::PtrStepSz<float4> img, const int &w, const int &h);

#endif
