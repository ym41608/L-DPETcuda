#ifndef GETPOSES_H
#define GETPOSES_H

#include <thrust/device_vector.h>

bool getPoses(thrust::device_vector<float4> *Poses4, thrust::device_vector<float2> *Poses2,
  thrust::device_vector<float> *Eas, float minEa, const float &delta, int *numPoses);

#endif