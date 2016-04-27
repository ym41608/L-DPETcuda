#include "DPT.h"

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "parameter.h"
#include "track.h"


using namespace cv;

void predictPose(pose *p, parameter *para) {
  pose tmp = *p;
  p->tx = tmp.tx + para->V0.tx + 0.5*para->A.tx;
  p->ty = tmp.ty + para->V0.ty + 0.5*para->A.ty;
  p->tz = tmp.tz + para->V0.tz + 0.5*para->A.tz;
  p->rx = tmp.rx + para->V0.rx + 0.5*para->A.rx;
  p->rz0 = tmp.rz0 + para->V0.rz0 + 0.5*para->A.rz0;
  p->rz1 = tmp.rz1 + para->V0.rz1 + 0.5*para->A.rz1;
  
  // update
  para->Prev = tmp;
}

void updatePrediction(pose *p, parameter *para) {
  // v
  para->V1 = para->V0;
  para->V0.tx  = p->tx  - (para->Prev).tx ;
  para->V0.ty  = p->ty  - (para->Prev).ty ;
  para->V0.tz  = p->tz  - (para->Prev).tz ;
  para->V0.rx  = p->rx  - (para->Prev).rx ;
  para->V0.rz0 = p->rz0 - (para->Prev).rz0;
  para->V0.rz1 = p->rz1 - (para->Prev).rz1;
  
  // a
  (para->A).tx  = (para->V0).tx  - (para->V1).tx ;
  (para->A).ty  = (para->V0).ty  - (para->V1).ty ;
  (para->A).tz  = (para->V0).tz  - (para->V1).tz ;
  (para->A).rx  = (para->V0).rx  - (para->V1).rx ;
  (para->A).rz0 = (para->V0).rz0 - (para->V1).rz0;
  (para->A).rz1 = (para->V0).rz1 - (para->V1).rz1;
}

void DPT(pose *p, parameter *para, const gpu::GpuMat &marker_d, const Mat &img, const bool &verbose) {
  
  // get img
  gpu::GpuMat img0(img);
  gpu::GpuMat img1(para->iDimY, para->iDimX, CV_32FC3);
  gpu::GpuMat img2(para->iDimY, para->iDimX, CV_32FC3);
  gpu::GpuMat img_d(para->iDimY, para->iDimX, CV_32FC4);
  img0.convertTo(img1, CV_32FC3, 1.0 / 255.0);
  gpu::cvtColor(img1, img2, CV_BGR2YCrCb);
  gpu::cvtColor(img2, img_d, CV_BGR2BGRA, 4);
  
  // get initial pose
  //predictPose(p, para);
  
  // tracking
  track(p, marker_d, img_d, para, verbose);
  
  // update velocity and accelaration
  //updatePrediction(p, para);
}