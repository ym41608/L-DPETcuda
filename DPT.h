#ifndef DPT_H
#define DPT_H

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "parameter.h"

using namespace cv;

void DPT(pose *p, parameter *para, const gpu::GpuMat &marker_d, const Mat &img, const bool &verbose);

#endif