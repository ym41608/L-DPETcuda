# L-DPETcuda

## testing direct pose estimation and tracking on video sequences

ver1. initial commit

ver1.1 fix Makefiel
  estimtation: 38 s
  tracking: 0.4 s (for SAMPLENUMT 900)

ver2. 3-scale block search
ver2.1 for be test case

ver3. Demo system
  FPS6~7
ver3,1
  fix redundant calculation for step rz0 rz1 * tz_mid

ver4 160504
  fix C2F fminf fmaxf
  fix idx <= numPoses
  fix Px, Py
  add cudaDeviceSyn to C2F and Track
  tracking: 0.09 s (for SAMPLENUMT 512)

ver5 160510
  makefile for tx1
  FPS 9~10 for tx1
  
ver5.1 160510
  fix Px Py to float
  tracking: 0.015 s (for SAMPLENUMT 512) on GTX770

ver6 160510
  intensity normalization factor tracking