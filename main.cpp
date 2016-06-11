#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <iostream>
#include <string>
#include "Time.h"
#include "parameter.h"
#include "APE.h"
#include "DPT.h"

using namespace std;
using namespace cv;

void drawCoordinate(Mat &img, float *ex_mat, float Sfx, float Sfy, float Px, float Py, const parameter &para) {
  const float minDim = (para.markerDimX > para.markerDimY) ? para.markerDimY : para.markerDimX;
  const float dimX = para.markerDimX;
  const float dimY = para.markerDimY;
  float factor = img.cols / float(para.iDimX);
  Sfx *= factor;
  Sfy *= factor;
  Px *= factor;
  Py *= factor;
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
   
  Point_<float> B1, B2, B3, B4, T1, T2, T3, T4;
  B1.x = (trans[0]*dimX + trans[1]*dimY + trans[2]*0.0 + trans[3]) / 
         (trans[8]*dimX + trans[9]*dimY + trans[10]*0.0 + trans[11]);
  B1.y = (trans[4]*dimX + trans[5]*dimY + trans[6]*0.0 + trans[7]) / 
         (trans[8]*dimX + trans[9]*dimY + trans[10]*0.0 + trans[11]);
  B2.x = (trans[0]*dimX + trans[1]*(-dimY) + trans[2]*0.0 + trans[3]) / 
         (trans[8]*dimX + trans[9]*(-dimY) + trans[10]*0.0 + trans[11]);
  B2.y = (trans[4]*dimX + trans[5]*(-dimY) + trans[6]*0.0 + trans[7]) / 
         (trans[8]*dimX + trans[9]*(-dimY) + trans[10]*0.0 + trans[11]);
  B3.x = (trans[0]*(-dimX) + trans[1]*(-dimY) + trans[2]*0.0 + trans[3]) / 
         (trans[8]*(-dimX) + trans[9]*(-dimY) + trans[10]*0.0 + trans[11]);
  B3.y = (trans[4]*(-dimX) + trans[5]*(-dimY) + trans[6]*0.0 + trans[7]) / 
         (trans[8]*(-dimX) + trans[9]*(-dimY) + trans[10]*0.0 + trans[11]);
  B4.x = (trans[0]*(-dimX) + trans[1]*dimY + trans[2]*0.0 + trans[3]) / 
         (trans[8]*(-dimX) + trans[9]*dimY + trans[10]*0.0 + trans[11]);
  B4.y = (trans[4]*(-dimX) + trans[5]*dimY + trans[6]*0.0 + trans[7]) / 
         (trans[8]*(-dimX) + trans[9]*dimY + trans[10]*0.0 + trans[11]);
  T1.x = (trans[0]*dimX + trans[1]*dimY + trans[2]*minDim + trans[3]) / 
         (trans[8]*dimX + trans[9]*dimY + trans[10]*minDim + trans[11]);
  T1.y = (trans[4]*dimX + trans[5]*dimY + trans[6]*minDim + trans[7]) / 
         (trans[8]*dimX + trans[9]*dimY + trans[10]*minDim + trans[11]);
  T2.x = (trans[0]*dimX + trans[1]*(-dimY) + trans[2]*minDim + trans[3]) / 
         (trans[8]*dimX + trans[9]*(-dimY) + trans[10]*minDim + trans[11]);
  T2.y = (trans[4]*dimX + trans[5]*(-dimY) + trans[6]*minDim + trans[7]) / 
         (trans[8]*dimX + trans[9]*(-dimY) + trans[10]*minDim + trans[11]);
  T3.x = (trans[0]*(-dimX) + trans[1]*(-dimY) + trans[2]*minDim + trans[3]) / 
         (trans[8]*(-dimX) + trans[9]*(-dimY) + trans[10]*minDim + trans[11]);
  T3.y = (trans[4]*(-dimX) + trans[5]*(-dimY) + trans[6]*minDim + trans[7]) / 
         (trans[8]*(-dimX) + trans[9]*(-dimY) + trans[10]*minDim + trans[11]);
  T4.x = (trans[0]*(-dimX) + trans[1]*dimY + trans[2]*minDim + trans[3]) / 
         (trans[8]*(-dimX) + trans[9]*dimY + trans[10]*minDim + trans[11]);
  T4.y = (trans[4]*(-dimX) + trans[5]*dimY + trans[6]*minDim + trans[7]) / 
         (trans[8]*(-dimX) + trans[9]*dimY + trans[10]*minDim + trans[11]);
  
  line(img, B1, B2, Scalar(255, 0, 0), 2, CV_AA);
  line(img, B2, B3, Scalar(255, 0, 0), 2, CV_AA);
  line(img, B3, B4, Scalar(255, 0, 0), 2, CV_AA);
  line(img, B4, B1, Scalar(255, 0, 0), 2, CV_AA);
  line(img, B1, T1, Scalar(0, 255, 0), 2, CV_AA);
  line(img, B2, T2, Scalar(0, 255, 0), 2, CV_AA);
  line(img, B3, T3, Scalar(0, 255, 0), 2, CV_AA);
  line(img, B4, T4, Scalar(0, 255, 0), 2, CV_AA);
  line(img, T1, T2, Scalar(255, 255, 0), 2, CV_AA);
  line(img, T2, T3, Scalar(255, 255, 0), 2, CV_AA);
  line(img, T3, T4, Scalar(255, 255, 0), 2, CV_AA);
  line(img, T4, T1, Scalar(255, 255, 0), 2, CV_AA);
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

int main(int argc, char *argv[]) {
  
  // parsing argument
  if (argc != 10) {
    cout << "invalid argument!" << endl;
    cout << "./DPET markerFile Sfx Sfy Px Py minMarkerDim minTz maxTz photometric" << endl;
    return -1;
  }
  
  // parse marker image
  Mat marker = cv::imread(argv[1]);
  if(!marker.data ) {
    cout <<  "Could not open marker " << argv[1] << endl ;
    return -1;
  }
  
  // assign parameters
  float Sfx = stof(string(argv[2]));
  float Sfy = stof(string(argv[3]));
  int Px = stoi(string(argv[4]));
  int Py = stoi(string(argv[5]));
  float minDim = stof(string(argv[6]));
  float minTz = stof(string(argv[7]));
  float maxTz = stof(string(argv[8]));
  bool photo = bool(stoi(string(argv[9])));
  
  // get camera
  VideoCapture cap(0);
  if (!cap.isOpened()) {
    cout <<  "Camera not found!" << endl ;
    return -1;
  }
  
  pose p;
  parameter para;
  Mat img, imgI;
  gpu::GpuMat marker_d;
  Timer time;
  int frameNum = 0;
  float *ex_mat = new float[12];
  namedWindow("DPET");
  
  // start!
  cout << "press e to start estimation, s to stop" << endl;
  while (frameNum > -1) {
    time.Reset(); time.Start();
    cap.read(img);
    resize(img, img, Size(640, 360));
    resize(img, imgI, Size(320, 180));
    
    // processing
    if (frameNum == 1) {
      cout << "estimating pose..." << endl;
      APE(&p, &para, marker, imgI, marker_d, Sfx, Sfy, Px, Py, minDim, minTz, maxTz, 0.25, photo, true);
      frameNum++;
    }
    else if (frameNum > 1){
      if (frameNum == 2)
        cout << "tracking pose!" << endl;
      DPT(&p, &para, marker_d, imgI, false);
      frameNum++;
    }
    
    // output to screen
    if (frameNum >= 1) {
      getExMat(ex_mat, p);
      drawCoordinate(img, ex_mat, Sfx, Sfy, Px, Py, para);
    }
    
    // calculate FPS
    time.Pause();
    char s[300];
    float FPS = 1000000 / float(time.get_count());
    sprintf(s, "Frame rate = %f", FPS);
    putText(img, s, Point(75, 75), FONT_HERSHEY_COMPLEX, 1, Scalar(255,255,0));
    if (frameNum == 2)
      imwrite("tmp.png", img);    

    // output img
    imshow("DPET", img);
    
    // waitKey
    char key = (char)waitKey(1);
    switch (key) {
      case 'e':
        frameNum = 1;
        break;
      case 's':
        frameNum = -1;
        break;
    }  
  }
  cout << "eunji get num. 1" << endl;
  delete[] ex_mat;
  return 0;
}
