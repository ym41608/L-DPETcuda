#ifndef EXPANDPOSES_H
#define EXPANDPOSES_H

#include <thrust/device_vector.h>
#include "parameter.h"

void expandPoses(thrust::device_vector<float4>* Poses4, thrust::device_vector<float2>* Poses2, const float& factor, parameter* para, int* numPoses);

#endif