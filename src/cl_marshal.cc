//===--- cl_marshal.cc - GPU in-place marshaling library        ----------===//
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
#include <cl.hpp>
#include "cl_marshal.h"
#include "embd.hpp"
extern "C" bool cl_aos_asta_bs(cl_command_queue cl_queue,
    cl_mem src, int height, int width,
    int tile_size) {
  clRetainMemObject(src);
  cl::CommandQueue queue = cl::CommandQueue(cl_queue);
  cl::Buffer buffer = cl::Buffer(src);
  cl::Context context;
  if(buffer.getInfo(CL_MEM_CONTEXT, &context) != CL_SUCCESS)
    return true;
  clRetainContext(context());
  clRetainCommandQueue(cl_queue);
  embd::file source_file("cl/cl_aos_asta.cl");
  std::istream &in = source_file.istream();
  std::string source_code(std::istreambuf_iterator<char>(in),
      (std::istreambuf_iterator<char>()));
  cl::Program::Sources source(1, std::make_pair(source_code.c_str(),
        source_code.length()+1));
  cl::Program program = cl::Program(context, source);
  std::vector<cl::Device> devices(0);
  cl_int err = program.build(devices);
  if (err != CL_SUCCESS)
    return true;
  cl::Kernel kernel(program, "BS_marshal");
  err = kernel.setArg(0, buffer);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(1, tile_size);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(2, width);
  if (err != CL_SUCCESS)
    return true;
  cl::NDRange global(height, width), local(tile_size, width);
  err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
  if (err != CL_SUCCESS)
    return true;
  return false;
}
#if 0
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
#endif