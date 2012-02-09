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
  PTTWAC_marshal<<<height/tile_size, NR_THREADS,
    ((tile_size*width+31)/32)*4>>>(src, tile_size, width, timer);
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

  PTTWAC_marshal_soa<<<height/tile_size*width, tile_size>>>(
      src, tile_size, width, finished, timer);
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    std::cerr << cudaGetErrorString(err) << std::endl;
  }
  cudaFree(finished);
  return cudaSuccess != err;
}
