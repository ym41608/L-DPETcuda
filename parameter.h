#ifndef PARAMETER_H
#define PARAMETER_H

// struct pose
struct pose {
  float tx, ty, tz, rx, rz0, rz1;
};

// class parameter
class parameter {
  public:
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
    float txV, tyV, tzV, rxV, rz0V, rz1V;
    float txA, tyZ, tzA, rxA, rz0A, rz1A;
};

#endif