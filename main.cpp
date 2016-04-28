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

void drawCoordinate(Mat & imgO, float *ex_mat, const float &Sfx, const float &Sfy, const float &Px, const float &Py, const parameter &para, const Mat &img) {
  float minDim = (para.markerDimX > para.markerDimY) ? para.markerDimY : para.markerDimX;
  imgO = img;
  
  float trans[16];
  trans[0] = Sfx*ex_mat[0] + Px*ex_mat[8];
	 trans[1] = Sfx*ex_mat[1] + Px*ex_mat[9];
	 trans[2] = Sfx*ex_mat[2] + Px*ex_mat[10];
	 trans[3] = Sfx*ex_mat[3] + Px*ex_mat[11];
	 trans[4] = (-Sfy)*ex_mat[4] + (Py-1)*ex_mat[8];
	 trans[5] = (-Sfy)*ex_mat[5] + (Py-1)*ex_mat[9];
	 trans[6] = (-Sfy)*ex_mat[6] + (Py-1)*ex_mat[10];
	 trans[7] = (-Sfy)*ex_mat[7]  + (Py-1)*ex_mat[11];
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
  Mat img, imgO;
  gpu::GpuMat marker_d;
  Timer time;
  int frameNum = 0;
  float *ex_mat = new float[12];
  namedWindow("DPET");
  
  // start!
  while (frameNum > -1) {
    time.Reset(); time.Start();
    if(!cap.read(img)) {
      cout << "no frame captured" << endl;
      break;
    }
    resize(img, img, Size(320, 180));
 
    // processing
    if (frameNum == 0)
      cout << "press any button" << endl;
    else if (frameNum == 1) {
      cout << "estimating pose..." << endl;
      APE(&p, &para, marker, img, marker_d, Sfx, Sfy, Px, Py, minDim, minTz, maxTz, 0.25, photo, true);
      frameNum++;
    }
    else {
      if (frameNum == 2)
        cout << "tracking pose!" << endl;
      DPT(&p, &para, marker_d, img, false);
      frameNum++;
    }
    
    // output to screen
    if (frameNum >= 1) {
      getExMat(ex_mat, p);
      drawCoordinate(imgO, ex_mat, Sfx, Sfy, Px, Py, para, img);
    }
    else
      imgO = img;
    
    // calculate FPS
    time.Pause();
    float FPS = 1000000 / float(time.get_count());
    imshow("DPET", imgO);
    if (frameNum == 2)
      imwrite("tmp.png", imgO);
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
