#ifndef PARAMETER_H
#define PARAMETER_H

// struct pose
struct pose {
  float tx, ty, tz, rx, rz0, rz1;
};

// class parameter
class parameter {
  public:
    parameter() {
      Prev.tx = 0;
      Prev.ty = 0;
      Prev.tz = 0;
      Prev.rx = 0;
      Prev.rz0 = 0;
      Prev.rz1 = 0;
      V0.tx = 0;
      V0.ty = 0;
      V0.tz = 0;
      V0.rx = 0;
      V0.rz0 = 0;
      V0.rz1 = 0;
      V1.tx = 0;
      V1.ty = 0;
      V1.tz = 0;
      V1.rx = 0;
      V1.rz0 = 0;
      V1.rz1 = 0;
      A.tx = 0;
      A.ty = 0;
      A.tz = 0;
      A.rx = 0;
      A.rz0 = 0;
      A.rz1 = 0;
    };
    void shrinkNet(const float& factor) {
      txS *= factor;
      tyS *= factor;
      tzS *= factor;
      rxS *= factor;
      rz0S *= factor;
      rz1S *= factor;
      delta *= factor;
    };
    
    // about pose net
    float delta;
    float tzMin, tzMax;
    float rxMin, rxMax;
    float rzMin, rzMax;
    float txS, tyS, tzS, rxS, rz0S, rz1S;
    
    // about camera
    float Sfx, Sfy;
    int Px, Py;
    
    // about marker
    int mDimX, mDimY;
    float markerDimX, markerDimY;
    
    // about img
    int iDimX, iDimY;
    double blur_sigma;
    
    // about photo
    bool photo;
    
    // about tracking
    pose Prev, V0, V1, A;
};

#endif