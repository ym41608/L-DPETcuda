#include "preCal.h"

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <iostream>
#include "parameter.h"
#include "preCal_kernel.h"
//#include <cuda_profiler_api.h>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace cv;
using namespace std;

double calSigmaValue(const gpu::GpuMat &marker_d, const parameter &para) {

  // convert to gray channel
  gpu::GpuMat marker_g(para.mDimX, para.mDimY, CV_32FC1);
  gpu::cvtColor(marker_d, marker_g, CV_BGR2GRAY);
  gpu::GpuMat tmp(para.mDimX, para.mDimY, CV_32FC1);
  
  // start calculation
  double blur_sigma = 1;
  float TVperNN = FLT_MAX;
  const float threshold = 1870 * (para.tzMin*para.tzMax) / (2*para.Sfx*para.markerDimX) / (2*para.Sfy*para.markerDimY) * (para.mDimX*para.mDimY);
  while (TVperNN > threshold) {
    blur_sigma++;
    int kSize = 4 * blur_sigma + 1;
    kSize = (kSize <= 32) ? kSize : 31;
    gpu::GaussianBlur(marker_g, tmp, Size(kSize, kSize), blur_sigma);
    TVperNN = getTVperNN(tmp, para.mDimX, para.mDimY);
  }
  return blur_sigma;
}

void preCal(parameter *para, gpu::GpuMat &marker_d, gpu::GpuMat &img_d, const Mat &marker, const Mat &img, 
            const float &Sfx, const float &Sfy, const float &Px, const float &Py, const float &minDim, const float &tzMin, const float &tzMax, 
            const float &delta, const bool &photo, const bool &verbose) {
  //cudaProfilerStart(); 
  // dim of images
  para->mDimX = marker.cols;
  para->mDimY = marker.rows;
  para->iDimX = img.cols;
  para->iDimY = img.rows;
  
  // intrinsic parameter
  para->Sfx = Sfx;
  para->Sfy = Sfy;
  para->Px = Px;
  para->Py = Py;
  
  // search range in pose domain
  float wm = float(marker.cols);
  float hm = float(marker.rows);
  float minmDim = fmin(hm, wm);
  para->markerDimX = wm / minmDim * minDim;
  para->markerDimY = hm / minmDim * minDim;
  float minmarkerDim = fmin(para->markerDimX, para->markerDimY);
  para->tzMin = tzMin;
  para->tzMax = tzMax;
  para->rxMin = 0.0;
  para->rxMax = 80*M_PI/180;
  para->rzMin = -M_PI;
  para->rzMax = M_PI;
  
  //bounds
  float m_tz = sqrt(tzMin*tzMax);
  float sqrt2 = sqrt(2);
  float invtmp = 1.0 / sqrt2 / m_tz;
  para->delta = delta;
  para->txS = delta*invtmp*2*(minDim);
  para->tyS = delta*invtmp*2*(minDim);
  para->tzS = delta*invtmp;
  para->rxS = delta*invtmp;
  para->rz0S = delta*sqrt2;
  para->rz1S = delta*sqrt2;
  
  // allocate memory to GPU
  gpu::GpuMat marker0(marker);
  gpu::GpuMat img0(img);
  gpu::GpuMat marker1(para->mDimY, para->mDimX, CV_32FC3);
  gpu::GpuMat img1(para->iDimY, para->iDimX, CV_32FC3);
  gpu::GpuMat marker2(para->mDimY, para->mDimX, CV_32FC3);
  gpu::GpuMat img2(para->iDimY, para->iDimX, CV_32FC3);
  marker0.convertTo(marker1, CV_32FC3, 1.0 / 255.0);
  img0.convertTo(img1, CV_32FC3, 1.0 / 255.0);

  // smooth images
  double blur_sigma = calSigmaValue(marker1, *para);
  para->blur_sigma = blur_sigma;
  if (verbose)
    std::cout << "blur sigma : " << blur_sigma << std::endl;
  int kSize = 4 * blur_sigma + 1;
  kSize = (kSize <= 32) ? kSize : 31;
  Ptr<gpu::FilterEngine_GPU> filter = gpu::createGaussianFilter_GPU(CV_32FC3, Size(kSize, kSize), blur_sigma);
  filter->apply(marker1, marker2, cv::Rect(0, 0, para->mDimX, para->mDimY));
  filter->apply(img1, img2, cv::Rect(0, 0, para->iDimX, para->iDimY));
  filter.release();
  
  // rgb2ycbcr
  gpu::cvtColor(marker2, marker_d, CV_BGR2YCrCb);
  gpu::cvtColor(img2, img1, CV_BGR2YCrCb);
  gpu::cvtColor(img1, img_d, CV_BGR2BGRA, 4);
  
  // photo
  para->photo = photo;
  
  //
  para->sigXoversigY = 1;
  para->offsetXY = 0;
}
