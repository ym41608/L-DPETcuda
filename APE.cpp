#include "APE.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "parameter.h"
#include "C2Festimate.h"
#include "preCal.h"
#include "Time.h"

using namespace cv;
using namespace std;

void APE(pose *p, parameter *para, const Mat &marker, const Mat &img, gpu::GpuMat &marker_d, const float &Sfx, const float &Sfy, const int &Px, const int &Py, 
         const float &minDim, const float &tzMin, const float &tzMax, const float &delta, const bool photo, const bool verbose) {
  
  // allocate
  Timer time;
  time.Reset(); time.Start();
  gpu::GpuMat img_d(img.rows, img.cols, CV_32FC4);
  
  // pre-calculation
  preCal(para, marker_d, img_d, marker, img, Sfx, Sfy, Px, Py, minDim, tzMin, tzMax, delta, photo, verbose);
  time.Pause();
  long long t1 = time.get_count();
  
  // C2Festimation
  time.Reset(); time.Start();
  C2Festimate(p, marker_d, img_d, para, verbose);
  time.Pause();
  cout << "estimate pre-time: " << t1 << " us." << endl;
  cout << "estimate post-time: " << time.get_count() << " us." << endl;
}
