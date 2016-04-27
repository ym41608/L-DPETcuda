#ifndef PRECAL_H
#define PRECAL_H

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "parameter.h"

using namespace cv;

void preCal(parameter *para, gpu::GpuMat &marker_d, gpu::GpuMat &img_d, const Mat &marker, const Mat &img, const float &Sfx, const float &Sfy, 
  const int &Px, const int &Py, const float &minDim, const float &tzMin, const float &tzMax,
  const float &delta, const bool &photo, const bool &verbose);

#endif