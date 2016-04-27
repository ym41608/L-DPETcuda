#ifndef DPT_H
#define DPT_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda_devptrs.hpp>
#include <vector_types.h>
#include "parameter.h"

using namespace cv;

void DPT(pose *p, parameter *para, const gpu::PtrStepSz<float3> &marker_d, const Mat &img, const bool &verbose);

#endif