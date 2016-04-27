#ifndef C2F_H
#define C2F_H

#include <opencv2/core/cuda_devptrs.hpp>
#include <vector_types.h>
#include "parameter.h"

using namespace cv;

void C2Festimate(pose *p, const gpu::PtrStepSz<float3> &marker_d, const gpu::PtrStepSz<float4> &img_d, parameter* para, const bool &verbose);

#endif
