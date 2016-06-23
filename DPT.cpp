#include "DPT.h"

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "parameter.h"
#include "track.h"


using namespace cv;

void DPT(pose *p, parameter *para, const gpu::GpuMat &marker_d, const Mat &img, const bool &verbose) {
  
  // get img
  gpu::GpuMat img0(img);
  gpu::GpuMat img1(para->iDimY, para->iDimX, CV_32FC3);
  gpu::GpuMat img2(para->iDimY, para->iDimX, CV_32FC3);
  gpu::GpuMat img_d(para->iDimY, para->iDimX, CV_32FC4);
  img0.convertTo(img1, CV_32FC3, 1.0 / 255.0);
  gpu::cvtColor(img1, img2, CV_BGR2YCrCb);
  gpu::cvtColor(img2, img_d, CV_BGR2BGRA, 4);
  
  // tracking
  track(p, marker_d, img_d, para, verbose);
}