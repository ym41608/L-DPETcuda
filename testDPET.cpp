#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <iostream>
#include "Time.h"
#include "parameter.h"
#include "APE.h"
#include "DPT.h"

#define SFX 3067.45 / 4
#define SFY 3067.45 / 4
#define PX 480.5
#define PY 270.5

using namespace std;
using namespace cv;

void drawCoordinate(Mat & imgO, float *ex_mat, const float &Sfx, const float &Sfy, const float &Px, const float &Py, const parameter &para, const Mat &img) {
  float minDim = (para.markerDimX > para.markerDimY) ? para.markerDimY : para.markerDimX;
  imgO = img;
  
  float trans[16];
  trans[0] = Sfx*ex_mat[0] + Px*ex_mat[8];
	 trans[1] = Sfx*ex_mat[1] + Px*ex_mat[9];
	 trans[2] = Sfx*ex_mat[2] + Px*ex_mat[10];
	 trans[3] = Sfx*ex_mat[3] + Px*ex_mat[11];
	 trans[4] = (-Sfy)*ex_mat[4] + Py*ex_mat[8];
	 trans[5] = (-Sfy)*ex_mat[5] + Py*ex_mat[9];
	 trans[6] = (-Sfy)*ex_mat[6] + Py*ex_mat[10];
	 trans[7] = (-Sfy)*ex_mat[7]  + Py*ex_mat[11];
	 trans[8] = ex_mat[8];
	 trans[9] = ex_mat[9];
	 trans[10] = ex_mat[10];
	 trans[11] = ex_mat[11];
	 trans[12] = 0;
	 trans[13] = 0;
	 trans[14] = 0;
	 trans[15] = 1;
   
  Point_<float> x, y, z, o;
  o.x = (trans[0]*0.0 + trans[1]*0.0 + trans[2]*0.0 + trans[3]) / 
        (trans[8]*0.0 + trans[9]*0.0 + trans[10]*0.0 + trans[11]);
  o.y = (trans[4]*0.0 + trans[5]*0.0 + trans[6]*0.0 + trans[7]) / 
        (trans[8]*0.0 + trans[9]*0.0 + trans[10]*0.0 + trans[11]);
  x.x = (trans[0]*minDim + trans[1]*0.0 + trans[2]*0.0 + trans[3]) / 
        (trans[8]*minDim + trans[9]*0.0 + trans[10]*0.0 + trans[11]);
  x.y = (trans[4]*minDim + trans[5]*0.0 + trans[6]*0.0 + trans[7]) / 
        (trans[8]*minDim + trans[9]*0.0 + trans[10]*0.0 + trans[11]);
  y.x = (trans[0]*0.0 + trans[1]*minDim + trans[2]*0.0 + trans[3]) / 
        (trans[8]*0.0 + trans[9]*minDim + trans[10]*0.0 + trans[11]);
  y.y = (trans[4]*0.0 + trans[5]*minDim + trans[6]*0.0 + trans[7]) / 
        (trans[8]*0.0 + trans[9]*minDim + trans[10]*0.0 + trans[11]);
  z.x = (trans[0]*0.0 + trans[1]*0.0 + trans[2]*minDim + trans[3]) / 
        (trans[8]*0.0 + trans[9]*0.0 + trans[10]*minDim + trans[11]);
  z.y = (trans[4]*0.0 + trans[5]*0.0 + trans[6]*minDim + trans[7]) / 
        (trans[8]*0.0 + trans[9]*0.0 + trans[10]*minDim + trans[11]);
  
  line(imgO, o, x, Scalar(255, 0, 0), 2, CV_AA);
  line(imgO, o, y, Scalar(0, 255, 0), 2, CV_AA);
  line(imgO, o, z, Scalar(0, 0, 255), 2, CV_AA);
}

void getExMat(float *ex_mat, const pose &P) {
  float rx = P.rx;
  float rz0 = P.rz0;
  float rz1 = P.rz1;
  ex_mat[0] =  cos(rz0)*cos(rz1) - sin(rz0)*cos(rx)*sin(rz1); //0
	ex_mat[1] = -cos(rz0)*sin(rz1) - sin(rz0)*cos(rx)*cos(rz1); //1
	ex_mat[2] = -(sin(rz0)*sin(rx)); //2
  ex_mat[3] = P.tx;
	ex_mat[4] =  sin(rz0)*cos(rz1) + cos(rz0)*cos(rx)*sin(rz1); //4
	ex_mat[5] = -sin(rz0)*sin(rz1) + cos(rz0)*cos(rx)*cos(rz1); //5
	ex_mat[6] = -(-cos(rz0)*sin(rx)); //6
  ex_mat[7] = P.ty;
	ex_mat[8] =  sin(rx)*sin(rz1); //8
	ex_mat[9] =  sin(rx)*cos(rz1); //9
	ex_mat[10] = -(cos(rx)); //10
  ex_mat[11] = P.tz;
}

int main() {
  Mat marker = cv::imread("img/be.png");
  if(!marker.data ) {
    cout <<  "Could not open marker" << endl ;
    return -1;
  }
  VideoCapture imgV("video/be_fm.avi");
  if (!imgV.isOpened()) {
    cout <<  "Could not open video" << endl ;
    return -1;
  }

  VideoWriter outputVideo;
  int ex = static_cast<int>(imgV.get(CV_CAP_PROP_FOURCC));
  Size S = Size((int) imgV.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
                (int) imgV.get(CV_CAP_PROP_FRAME_HEIGHT));
  outputVideo.open("result/result.avi", ex, imgV.get(CV_CAP_PROP_FPS), S, true);
  if (!outputVideo.isOpened()) {
    cout <<  "Could not open output video" << endl ;
    return -1;
  }
  
  pose p;
  parameter para;
  Mat img, imgO;
  gpu::GpuMat marker_d, img_d;
  Timer time;
  long long totolTime = 0;
  float *ex_mat = new float[12];
  
  // initialize
  imgV.read(img);
  APE(&p, &para, marker, img, marker_d, SFX, SFY, PX, PY, 0.25*3300/12350.0, 0.3, 1, 0.25, false, true);
  getExMat(ex_mat, p);
  drawCoordinate(imgO, ex_mat, SFX, SFY, PX, PY, para, img);
  imwrite("result/result.png", imgO);
  outputVideo.write(imgO);
  
  unsigned int count = 1;
  // tracking
  while(imgV.read(img)) {
    count++;
    time.Reset(); time.Start();
    DPT(&p, &para, marker_d, img, false);
    getExMat(ex_mat, p);
    drawCoordinate(imgO, ex_mat, SFX, SFY, PX, PY, para, img);
    time.Pause();
    outputVideo.write(imgO);
    totolTime += time.get_count();
    cout << count << endl;
    //if (count == 210)
    //  break;
  }
  float avgTime = totolTime / float(count);
  cout << "avg time: " << avgTime << endl;
  delete[] ex_mat;
  return 0;
}
