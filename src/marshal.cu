//===--- marshal.cu - GPU in-place marshaling library          ----------===//
// (C) Copyright 2012 The Board of Trustees of the University of Illinois.
// All rights reserved.
//
//                            libmarshal
// Developed by:
//                           IMPACT Research Group
//                  University of Illinois, Urbana-Champaign
// 
// This file is distributed under the Illinois Open Source License.
// See LICENSE.TXT for details.
//
// Author: I-Jui Sung (sung10@illinois.edu)
//
//===---------------------------------------------------------------------===//
//
//  This file defines the interface of the libmarshal 
//
//===---------------------------------------------------------------------===//

#include <cstdlib>
#include <cassert>
#include <iostream>
#include "marshal.h"
#include "marshal_kernel.cu"
extern "C" bool gpu_aos_asta_bs(float *src, int height, int width,
    int tile_size, clock_t *timer) {
  assert ((height/tile_size)*tile_size == height);
  dim3 threads (width, tile_size, 1);
  BS_marshal<<<height/tile_size, threads>>>(src, tile_size, width, timer);
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    std::cerr << cudaGetErrorString(err) << std::endl;
  }
  return cudaSuccess != err;
}

#define NR_THREADS 64
extern "C" bool gpu_aos_asta_pttwac(float *src, int height, int width,
    int tile_size, clock_t *timer) {
  assert ((height/tile_size)*tile_size == height);
  PTTWAC_marshal<<<min(height/tile_size,1024), NR_THREADS,
    ((tile_size*width+31)/32)*4>>>(src, height, tile_size, width, timer);
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    std::cerr << cudaGetErrorString(err) << std::endl;
  }
  return cudaSuccess != err;
}

extern "C" bool gpu_soa_asta_pttwac(float *src, int height, int width,
    int tile_size, clock_t *timer) {
  assert ((height/tile_size)*tile_size == height);
  int *finished;
  cudaMalloc(&finished, height*width/tile_size*sizeof(int));
  cudaMemset(finished, 0, height*width/tile_size*sizeof(int));

  size_t grid = min(height/tile_size*width, 1024);
  PTTWAC_marshal_soa<<<grid, tile_size>>>(
      src, height, tile_size, width, finished, timer);
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    std::cerr << cudaGetErrorString(err) << std::endl;
  }
  cudaFree(finished);
  return cudaSuccess != err;
}
