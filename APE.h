#ifndef APE_H
#define APE_H

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "parameter.h"

using namespace cv;

void APE(pose *p, parameter *para, const Mat &marker, const Mat &img, gpu::GpuMat &marker_d, const float &Sfx, const float &Sfy, const float &Px, const float &Py, 
         const float &minDim, const float &tzMin, const float &tzMax, const float &delta, const bool photo, const bool verbose);

#endif