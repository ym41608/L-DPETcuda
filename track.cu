#include "track.h"

#include <opencv2/core/cuda_devptrs.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <iostream>
#include "parameter.h"
#include "device_common.h"
//#include <cuda_profiler_api.h>

using namespace cv;
using namespace std;

// constant
__constant__ float2 const_Mcoor[SAMPLE_NUMT];
__constant__ float4 const_marker[SAMPLE_NUMT];

// texture
texture<float4, cudaTextureType2D, cudaReadModeElementType> tex_imgYCrCb;

//{ --- for random pixels generation --- //
struct intwhprgT {
  int w, h;
  
  __host__ __device__
  intwhprgT(int _w = 0, int _h = 100) {
    w = _w;
    h = _h;
  };
  __host__ __device__
  int2 operator()(const int n) const {
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> distw(-1, w - 1);
    thrust::uniform_int_distribution<int> disth(-1, h - 1);
    rng.discard(n);
    return make_int2(distw(rng), disth(rng));
  };
};

__global__ void assign_kernelT(float2 *mCoor, float4 *mValue, int2 *rand_coor, const gpu::PtrStepSz<float3> marker_dptr, const int2 mDim, const float2 markerDim) {
  const int tIdx = threadIdx.x;
  const int Idx = blockIdx.x * BLOCK_SIZE + tIdx;
  if (Idx >= SAMPLE_NUMT)
    return;
  
  int x = rand_coor[Idx].x;
  int y = rand_coor[Idx].y;
  
  float3 v = marker_dptr(y, x);
  mValue[Idx] = make_float4(v.x, v.y, v.z, 0);
  
  float2 coor;
  coor.x = (2 * float(x) - mDim.x) / mDim.x * markerDim.x;
  coor.y = -(2 * float(y) - mDim.y) / mDim.y * markerDim.y;
  mCoor[Idx] = coor;
}

void randSampleT(thrust::device_vector<float2>* mCoor, thrust::device_vector<float4>* mValue, const gpu::PtrStepSz<float3> &marker_d, const int2 &mDim, const float2 &markerDim) {

  // rand pixel
  thrust::device_vector<int2> rand_coor(SAMPLE_NUMT, make_int2(0, 0));
  thrust::counting_iterator<int> i0(509);
  thrust::transform(i0, i0 + SAMPLE_NUMT, rand_coor.begin(), intwhprgT(mDim.x, mDim.y));

  // get pixel value and position
  const int BLOCK_NUM = (SAMPLE_NUMT - 1) / BLOCK_SIZE + 1;
  assign_kernelT << < BLOCK_NUM, BLOCK_SIZE >> > (thrust::raw_pointer_cast(mCoor->data()), thrust::raw_pointer_cast(mValue->data()), thrust::raw_pointer_cast(rand_coor.data()), marker_d, mDim, markerDim);
  
  // bind to const mem
  cudaMemcpyToSymbol(const_Mcoor, thrust::raw_pointer_cast(mCoor->data()), sizeof(float2)* SAMPLE_NUMT, 0, cudaMemcpyDeviceToDevice);
  cudaMemcpyToSymbol(const_marker, thrust::raw_pointer_cast(mValue->data()), sizeof(float4)* SAMPLE_NUMT, 0, cudaMemcpyDeviceToDevice);
}
//}

//{ --- for create pose set --- //
__global__
void createSet_kernelT(float4* Poses4, float2* Poses2, const float tx, const float ty, const float tz, const float rx, const float rz0, const float rz1) {
  const int tIdx = threadIdx.x;
  const int Idx = blockIdx.x * BLOCK_SIZE + tIdx;
  
  if (Idx >= 262144)
    return;
    
  const int idrz1 = Idx % 8;
  const int idrz0 = (Idx / 8) % 8;
  const int idrx = (Idx / 64) % 8;
  const int idtz = (Idx / 512) % 8;
  const int idty = (Idx / 4096) % 8;
  const int idtx = (Idx / 32768) % 8;
  
  float4 p4;
  float2 p2;
  //p4.x = tx - 0.012 + idtx*0.004;
  //p4.y = ty - 0.012 + idty*0.004;
  //p4.z = tz - 0.024 + idtz*0.008;
  //p4.w = rx - 0.036 + idrx*0.012;
  //p2.x = rz0 - 0.06 + idrz0*0.02;
  //p2.y = rz1 - 0.06 + idrz1*0.02;
  p4.x = tx - 0.006 + idtx*0.002;
  p4.y = ty - 0.006 + idty*0.002;
  p4.z = tz - 0.012 + idtz*0.004;
  p4.w = rx - 0.018 + idrx*0.006;
  p2.x = rz0 - 0.03 + idrz0*0.01;
  p2.y = rz1 - 0.03 + idrz1*0.01;
  
  Poses4[Idx] = p4;
  Poses2[Idx] = p2;			
}

void createSetT(thrust::device_vector<float4> *Poses4, thrust::device_vector<float2> *Poses2, const pose &pini) {
    // tracking pattern
  const int BLOCK_NUM = 262143 / BLOCK_SIZE + 1;
  createSet_kernelT <<< BLOCK_NUM, BLOCK_SIZE >>> (thrust::raw_pointer_cast(Poses4->data()), thrust::raw_pointer_cast(Poses2->data()), 
    pini.tx, pini.ty, pini.tz, pini.rx, pini.rz0, pini.rz1);
}
//}

//{ --- for calculate Ea --- //
__global__
void calEa_NP_kernelT(float4 *Poses4, float2 *Poses2, float *Eas, const float2 Sf, const int2 P, const float2 normDim, const int2 imgDim) {
  const int tIdx = threadIdx.x;
  const int Idx = blockIdx.x * BLOCK_SIZE + tIdx;

  if (Idx >= 262144)
    return;

  // calculate transformation
  float tx, ty, tz, rx, rz0, rz1;
  float rz0Cos, rz0Sin, rz1Cos, rz1Sin, rxCos, rxSin;
  float t0, t1, t3, t4, t5, t7, t8, t9, t11;
  float r11, r12, r21, r22, r31, r32;

  // get pose parameter
  tx = Poses4[Idx].x;
  ty = Poses4[Idx].y;
  tz = Poses4[Idx].z;
  rx = Poses4[Idx].w;
  rz0 = Poses2[Idx].x;
  rz1 = Poses2[Idx].y;

  rz0Cos = cosf(rz0); rz0Sin = sinf(rz0);
  rz1Cos = cosf(rz1); rz1Sin = sinf(rz1);
  rxCos = cosf(rx); rxSin = sinf(rx);

  //  z coordinate is y cross x   so add minus
  r11 = rz0Cos * rz1Cos - rz0Sin * rxCos * rz1Sin;
  r12 = -rz0Cos * rz1Sin - rz0Sin * rxCos * rz1Cos;
  r21 = rz0Sin * rz1Cos + rz0Cos * rxCos * rz1Sin;
  r22 = -rz0Sin * rz1Sin + rz0Cos * rxCos * rz1Cos;
  r31 = rxSin * rz1Sin;
  r32 = rxSin * rz1Cos;

  // final transfomration
  t0 = Sf.x*r11 + P.x*r31;
  t1 = Sf.x*r12 + P.x*r32;
  t3 = Sf.x*tx + P.x*tz;
  t4 = (-Sf.y)*r21 + (P.y - 1)*r31;
  t5 = (-Sf.y)*r22 + (P.y - 1)*r32;
  t7 = (-Sf.y)*ty + (P.y - 1)*tz;
  t8 = r31;
  t9 = r32;
  t11 = tz;

  // reject transformations make marker out of boundary
  float invc1z = 1 / (t8*(-normDim.x) + t9*(-normDim.y) + t11);
  float c1x = (t0*(-normDim.x) + t1*(-normDim.y) + t3) * invc1z;
  float c1y = (t4*(-normDim.x) + t5*(-normDim.y) + t7) * invc1z;
  float invc2z = 1 / (t8*(+normDim.x) + t9*(-normDim.y) + t11);
  float c2x = (t0*(+normDim.x) + t1*(-normDim.y) + t3) * invc2z;
  float c2y = (t4*(+normDim.x) + t5*(-normDim.y) + t7) * invc2z;
  float invc3z = 1 / (t8*(+normDim.x) + t9*(+normDim.y) + t11);
  float c3x = (t0*(+normDim.x) + t1*(+normDim.y) + t3) * invc3z;
  float c3y = (t4*(+normDim.x) + t5*(+normDim.y) + t7) * invc3z;
  float invc4z = 1 / (t8*(-normDim.x) + t9*(+normDim.y) + t11);
  float c4x = (t0*(-normDim.x) + t1*(+normDim.y) + t3) * invc4z;
  float c4y = (t4*(-normDim.x) + t5*(+normDim.y) + t7) * invc4z;
  float minx = min(c1x, min(c2x, min(c3x, c4x)));
  float maxx = max(c1x, max(c2x, max(c3x, c4x)));
  float miny = min(c1y, min(c2y, min(c3y, c4y)));
  float maxy = max(c1y, max(c2y, max(c3y, c4y)));
  if ((minx < 0) | (maxx >= imgDim.x) | (miny < 0) | (maxy >= imgDim.y)) {
    Eas[Idx] = 100.0;
    return;
  }

  // calculate Ea
  float score = 0.0;
  float invz;
  float4 YCrCb_tex, YCrCb_const;
  float u, v;
  for (int i = 0; i < SAMPLE_NUMT; i++) {

    // calculate coordinate on camera image
    invz = 1 / (t8*const_Mcoor[i].x + t9*const_Mcoor[i].y + t11);
    u = (t0*const_Mcoor[i].x + t1*const_Mcoor[i].y + t3) * invz;
    v = (t4*const_Mcoor[i].x + t5*const_Mcoor[i].y + t7) * invz;

    // get value from constmem
    YCrCb_const = const_marker[i];

    // get value from texture
    YCrCb_tex = tex2D(tex_imgYCrCb, u, v);

    // calculate distant
    score += (2.852 * abs(YCrCb_tex.x - YCrCb_const.x) + abs(YCrCb_tex.y - YCrCb_const.y) + 1.264 * abs(YCrCb_tex.z - YCrCb_const.z));
  }
  Eas[Idx] = score / (SAMPLE_NUMT * 5.116);
}

__global__
void calEa_P_kernelT(float4 *Poses4, float2 *Poses2, float *Eas, const float2 Sf, const int2 P, const float2 normDim, const int2 imgDim) {
  const int tIdx = threadIdx.x;
  const int Idx = blockIdx.x * BLOCK_SIZE + tIdx;

  if (Idx >= 262144)
    return;

  // calculate transformation
  float tx, ty, tz, rx, rz0, rz1;
  float rz0Cos, rz0Sin, rz1Cos, rz1Sin, rxCos, rxSin;
  float t0, t1, t3, t4, t5, t7, t8, t9, t11;
  float r11, r12, r21, r22, r31, r32;

  // get pose parameter
  tx = Poses4[Idx].x;
  ty = Poses4[Idx].y;
  tz = Poses4[Idx].z;
  rx = Poses4[Idx].w;
  rz0 = Poses2[Idx].x;
  rz1 = Poses2[Idx].y;

  rz0Cos = cosf(rz0); rz0Sin = sinf(rz0);
  rz1Cos = cosf(rz1); rz1Sin = sinf(rz1);
  rxCos = cosf(rx); rxSin = sinf(rx);

  //  z coordinate is y cross x   so add minus
  r11 = rz0Cos * rz1Cos - rz0Sin * rxCos * rz1Sin;
  r12 = -rz0Cos * rz1Sin - rz0Sin * rxCos * rz1Cos;
  r21 = rz0Sin * rz1Cos + rz0Cos * rxCos * rz1Sin;
  r22 = -rz0Sin * rz1Sin + rz0Cos * rxCos * rz1Cos;
  r31 = rxSin * rz1Sin;
  r32 = rxSin * rz1Cos;

  // final transfomration
  t0 = Sf.x*r11 + P.x*r31;
  t1 = Sf.x*r12 + P.x*r32;
  t3 = Sf.x*tx + P.x*tz;
  t4 = (-Sf.y)*r21 + (P.y - 1)*r31;
  t5 = (-Sf.y)*r22 + (P.y - 1)*r32;
  t7 = (-Sf.y)*ty + (P.y - 1)*tz;
  t8 = r31;
  t9 = r32;
  t11 = tz;

  // reject transformations make marker out of boundary
  float invc1z = 1 / (t8*(-normDim.x) + t9*(-normDim.y) + t11);
  float c1x = (t0*(-normDim.x) + t1*(-normDim.y) + t3) * invc1z;
  float c1y = (t4*(-normDim.x) + t5*(-normDim.y) + t7) * invc1z;
  float invc2z = 1 / (t8*(+normDim.x) + t9*(-normDim.y) + t11);
  float c2x = (t0*(+normDim.x) + t1*(-normDim.y) + t3) * invc2z;
  float c2y = (t4*(+normDim.x) + t5*(-normDim.y) + t7) * invc2z;
  float invc3z = 1 / (t8*(+normDim.x) + t9*(+normDim.y) + t11);
  float c3x = (t0*(+normDim.x) + t1*(+normDim.y) + t3) * invc3z;
  float c3y = (t4*(+normDim.x) + t5*(+normDim.y) + t7) * invc3z;
  float invc4z = 1 / (t8*(-normDim.x) + t9*(+normDim.y) + t11);
  float c4x = (t0*(-normDim.x) + t1*(+normDim.y) + t3) * invc4z;
  float c4y = (t4*(-normDim.x) + t5*(+normDim.y) + t7) * invc4z;
  float minx = min(c1x, min(c2x, min(c3x, c4x)));
  float maxx = max(c1x, max(c2x, max(c3x, c4x)));
  float miny = min(c1y, min(c2y, min(c3y, c4y)));
  float maxy = max(c1y, max(c2y, max(c3y, c4y)));
  if ((minx < 0) | (maxx >= imgDim.x) | (miny < 0) | (maxy >= imgDim.y)) {
    Eas[Idx] = 100.0;
    return;
  }

  // calculate Ea
  float score = 0.0;
  float invz;
  float4 YCrCb_tex, YCrCb_const;
  float u, v;

  // parameters for normalization
  float sumXi = 0; float sumYi = 0;
  float sumXiSqrd = 0; float sumYiSqrd = 0;
  float Xi, Yi;
  //float4 tmpp[SAMPLE_NUMT];
  for (int i = 0; i < SAMPLE_NUMT; i++) {
    // calculate coordinate on camera image
    invz = 1 / (t8*const_Mcoor[i].x + t9*const_Mcoor[i].y + t11);
    u = (t0*const_Mcoor[i].x + t1*const_Mcoor[i].y + t3) * invz;
    v = (t4*const_Mcoor[i].x + t5*const_Mcoor[i].y + t7) * invz;

    // get value from constmem
    YCrCb_const = const_marker[i];

    // get value from texture
    YCrCb_tex = tex2D(tex_imgYCrCb, u, v);

    // accumulation for normalization
    Xi = YCrCb_const.x;
    Yi = YCrCb_tex.x;
    sumXi += Xi;
    sumYi += Yi;
    sumXiSqrd += (Xi*Xi);
    sumYiSqrd += (Yi*Yi);
    //tmpp[i] = YCrCb_tex;
  }

  // normalization parameter
  float sigX = sqrt((sumXiSqrd - (sumXi*sumXi) / SAMPLE_NUMT) / SAMPLE_NUMT) + 0.0000001;
  float sigY = sqrt((sumYiSqrd - (sumYi*sumYi) / SAMPLE_NUMT) / SAMPLE_NUMT) + 0.0000001;
  float meanX = sumXi / SAMPLE_NUMT;
  float meanY = sumYi / SAMPLE_NUMT;
  float sigXoversigY = sigX / sigY;
  float faster = -meanX + sigXoversigY*meanY;

  //for (int i = 0; i < SAMPLE_NUMT; i++) {
  //  YCrCb_const = const_marker[i];
  //  YCrCb_tex = tmpp[i];
  //  score += (2.852*abs(YCrCb_const.x - sigXoversigY*YCrCb_tex.x + faster) + abs(YCrCb_tex.y - YCrCb_const.y) + 1.264*abs(YCrCb_tex.z - YCrCb_const.z));
  //}
  for (int i = 0; i < SAMPLE_NUMT; i++) {
    // calculate coordinate on camera image
    invz = 1 / (t8*const_Mcoor[i].x + t9*const_Mcoor[i].y + t11);
    u = (t0*const_Mcoor[i].x + t1*const_Mcoor[i].y + t3) * invz;
    v = (t4*const_Mcoor[i].x + t5*const_Mcoor[i].y + t7) * invz;

    // get value from constmem
    YCrCb_const = const_marker[i];

    // get value from texture
    YCrCb_tex = tex2D(tex_imgYCrCb, u, v);
    score += (2.852*abs(YCrCb_const.x - sigXoversigY*YCrCb_tex.x + faster) + abs(YCrCb_tex.y - YCrCb_const.y) + 1.264*abs(YCrCb_tex.z - YCrCb_const.z));
  }
  Eas[Idx] = score / (SAMPLE_NUMT * 5.116);
}
  
void calEaT(thrust::device_vector<float4> *Poses4, thrust::device_vector<float2> *Poses2, thrust::device_vector<float> *Eas, 
    const float2 &Sf, const int2 &P, const float2 &markerDim, const int2 &iDim, const bool &photo) {
  const int BLOCK_NUM = 262143 / BLOCK_SIZE + 1;
  if (photo) {
    calEa_P_kernelT << < BLOCK_NUM, BLOCK_SIZE >> > (thrust::raw_pointer_cast(Poses4->data()), thrust::raw_pointer_cast(Poses2->data()), 
      thrust::raw_pointer_cast(Eas->data()), Sf, P, markerDim, iDim);
  }
  else {
    calEa_NP_kernelT << < BLOCK_NUM, BLOCK_SIZE >> > (thrust::raw_pointer_cast(Poses4->data()), thrust::raw_pointer_cast(Poses2->data()), 
      thrust::raw_pointer_cast(Eas->data()), Sf, P, markerDim, iDim);
  }
}
//}

//{ --- for Track --- //
void track(pose *p, const gpu::PtrStepSz<float3> &marker_d, const gpu::PtrStepSz<float4> &img_d, parameter *para, const bool &verbose) {
  // bind texture memory
  tex_imgYCrCb.addressMode[0] = cudaAddressModeBorder;
  tex_imgYCrCb.addressMode[1] = cudaAddressModeBorder;
  tex_imgYCrCb.filterMode = cudaFilterModePoint;
  tex_imgYCrCb.normalized = false;
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
  cudaBindTexture2D(0, &tex_imgYCrCb, img_d.data, &desc, para->iDimX, para->iDimY, img_d.step);
  
  // allocate sample memory
  thrust::device_vector<float2> mCoor(SAMPLE_NUMT, make_float2(0, 0));
  thrust::device_vector<float4> mValue(SAMPLE_NUMT, make_float4(0, 0, 0, 0));
  randSampleT(&mCoor, &mValue, marker_d, make_int2(para->mDimX, para->mDimY), make_float2(para->markerDimX, para->markerDimY));

  // initialize the net
  thrust::device_vector<float4> Poses4(262144);
  thrust::device_vector<float2> Poses2(262144);
  thrust::device_vector<float> Eas(262144);
  createSetT(&Poses4, &Poses2, *p);
  
  // calEa
  calEaT(&Poses4, &Poses2, &Eas, make_float2(para->Sfx, para->Sfy), make_int2(para->Px, para->Py), 
    make_float2(para->markerDimX, para->markerDimY), make_int2(para->iDimX, para->iDimY), para->photo);
    
  // findMin
  thrust::device_vector<float>::iterator iter = thrust::min_element(Eas.begin(), Eas.end());
  float bestEa = *iter;
  if (verbose)
    std::cout << "$$$ bestEa = " << bestEa << endl;
  const int idx = iter - Eas.begin();
  float4 p4 = Poses4[idx];
  float2 p2 = Poses2[idx];
  p->tx = p4.x;
  p->ty = p4.y;
  p->tz = p4.z;
  p->rx = p4.w;
  p->rz0 = p2.x;
  p->rz1 = p2.y;
  
  // unbind the texture
  cudaUnbindTexture (&tex_imgYCrCb);
  return;
}
//}