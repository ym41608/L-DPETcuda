CC=g++
NVCC=nvcc
CFLAGS=-std=c++11 -O3
NVCCFLAGS=-O3 -arch=sm_53
INCS=-I/usr/local/cuda/include
LIBS=-L/usr/local/cuda/lib -lcudart

all: DPET

DPET: main.o APE.o preCal.o preCal_kernel.o C2Festimate.o getPoses.o expandPoses.o DPT.o track.o
	$(CC) -o DPET main.o APE.o preCal.o preCal_kernel.o C2Festimate.o getPoses.o expandPoses.o DPT.o track.o `pkg-config --libs opencv` $(LIBS)

main.o: main.cpp APE.h DPT.h Time.h parameter.h
	$(CC) -c $(CFLAGS) main.cpp `pkg-config --cflags opencv` $(INCS)

testDPET: testDPET.o APE.o preCal.o preCal_kernel.o C2Festimate.o getPoses.o expandPoses.o DPT.o track.o
	$(CC) -o testDPET testDPET.o APE.o preCal.o preCal_kernel.o C2Festimate.o getPoses.o expandPoses.o DPT.o track.o `pkg-config --libs opencv` $(LIBS)

testDPET.o: testDPET.cpp APE.h DPT.h Time.h parameter.h
	$(CC) -c $(CFLAGS) testDPET.cpp `pkg-config --cflags opencv` $(INCS)

APE.o: APE.cpp APE.h Time.h parameter.h preCal.h C2Festimate.h
	$(CC) -c $(CFLAGS) APE.cpp `pkg-config --cflags opencv` $(INCS)

DPT.o: DPT.cpp DPT.h parameter.h track.h
	$(CC) -c $(CFLAGS) DPT.cpp `pkg-config --cflags opencv` $(INCS)
 
preCal.o: preCal.cpp preCal.h parameter.h preCal_kernel.h
	$(CC) -c $(CFLAGS) preCal.cpp $(INCS)

preCal_kernel.o: preCal_kernel.cu preCal_kernel.h device_common.h
	$(NVCC) -c $(NVCCFLAGS) preCal_kernel.cu

C2Festimate.o: C2Festimate.cu C2Festimate.h parameter.h device_common.h getPoses.h expandPoses.h
	$(NVCC) -c $(NVCCFLAGS) C2Festimate.cu

getPoses.o: getPoses.cu getPoses.h device_common.h
	$(NVCC) -c $(NVCCFLAGS) getPoses.cu

expandPoses.o: expandPoses.cu expandPoses.h parameter.h device_common.h
	$(NVCC) -c $(NVCCFLAGS) expandPoses.cu
  
track.o: track.cu track.h parameter.h device_common.h
	$(NVCC) -c $(NVCCFLAGS) track.cu

clean: 
	rm -rf *.o DPET testDPET
