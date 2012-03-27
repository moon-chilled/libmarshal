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
#include "cl_profile.h"
#include "cl_marshal.h"
#include "local_cl.hpp"
#include "embd.hpp"
#include "singleton.hpp"
namespace {
class MarshalProg {
 public:
  ~MarshalProg() {}
  MarshalProg(void): program(NULL), context_(NULL) {
    libmarshal::file source_file("cl/cl_aos_asta.cl");
    std::istream &in = source_file.istream();
    source_code_ = std::string(std::istreambuf_iterator<char>(in),
	(std::istreambuf_iterator<char>()));
    source_ = cl::Program::Sources(1,
      std::make_pair(source_code_.c_str(), source_code_.length()+1));
  }
  cl_uint GetCtxRef(void) const {
    cl_uint rc = 0;
    if (context_) {
      clGetContextInfo(context_,
          CL_CONTEXT_REFERENCE_COUNT, sizeof(cl_uint), &rc, NULL);
    }
    return rc;
  }
  bool Init(cl_context clcontext) {
    if (context_ != clcontext) { //Trigger recompilation
      context_ = clcontext;
      cl::Context context(clcontext);
      clRetainContext(clcontext);
      cl_uint old_ref = GetCtxRef();
      program = cl::Program(context, source_);
      if (CL_SUCCESS != program.build())
        return true;
      // On Apple and ATI, build a program has an implied clRetainContext.
      // To avoid leak, release the additional lock. Note: Not thread-safe
      if (old_ref != GetCtxRef())
        clReleaseContext(clcontext);
    }
    return false;
  }

  cl::Program program;
 private:
  cl::Program::Sources source_;
  std::string source_code_;
  cl_context context_;
};
typedef Singleton<MarshalProg> MarshalProgSingleton;
}

#define NR_THREADS 256
#define IS_POW2(x) (x && !(x &( x- 1)))
//#define IS_POW2(x) 0
// v: 32-bit word input to count zero bits on right
static int count_zero_bits(unsigned int v) {
  unsigned int c = 32; // c will be the number of zero bits on the right
  v &= -signed(v);
  if (v) c--;
  if (v & 0x0000FFFF) c -= 16;
  if (v & 0x00FF00FF) c -= 8;
  if (v & 0x0F0F0F0F) c -= 4;
  if (v & 0x33333333) c -= 2;
  if (v & 0x55555555) c -= 1;
  return c;
}
extern "C" bool cl_aos_asta_bs(cl_command_queue cl_queue,
    cl_mem src, int height, int width,
    int tile_size) {
  // Standard preparation of invoking a kernel
  cl::CommandQueue queue = cl::CommandQueue(cl_queue);
  clRetainCommandQueue(cl_queue);
  Profiling prof(queue, "AOS-ASTA BS");
  cl::Buffer buffer = cl::Buffer(src);
  clRetainMemObject(src);
  cl::Context context;
  if(buffer.getInfo(CL_MEM_CONTEXT, &context) != CL_SUCCESS)
    return true;
  MarshalProg *marshalprog = MarshalProgSingleton::Instance();
  marshalprog->Init(context());

  cl::Kernel kernel(marshalprog->program, 
      IS_POW2(tile_size) ? "BS_marshal_power2":"BS_marshal");
  if (CL_SUCCESS != kernel.setArg(0, buffer))
    return true;
  cl_int err = kernel.setArg(1,
      IS_POW2(tile_size)? count_zero_bits(tile_size) : tile_size);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(2, width);
  err |= kernel.setArg(3, width*(tile_size+1)*sizeof(cl_float), NULL);
  if (err != CL_SUCCESS)
    return true;
  cl::NDRange global(height/tile_size*NR_THREADS), local(NR_THREADS);
  err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, NULL,
    prof.GetEvent());
  if (err != CL_SUCCESS)
    return true;
#ifdef LIBMARSHAL_OCL_PROFILE
  prof.Report(height*width*sizeof(float)*2);
#endif
  return false;
}

extern "C" bool cl_aos_asta_pttwac(cl_command_queue cl_queue,
    cl_mem src, int height, int width, int tile_size) {
  // Standard preparation of invoking a kernel
  cl::CommandQueue queue = cl::CommandQueue(cl_queue);
  clRetainCommandQueue(cl_queue);
  Profiling prof(queue, "AOS-ASTA PTTWAC");
  cl::Buffer buffer = cl::Buffer(src);
  clRetainMemObject(src);
  cl::Context context;
  if(buffer.getInfo(CL_MEM_CONTEXT, &context) != CL_SUCCESS)
    return true;
  MarshalProg *marshalprog = MarshalProgSingleton::Instance();
  marshalprog->Init(context());

  assert ((height/tile_size)*tile_size == height);
  cl::Kernel kernel(marshalprog->program, "PTTWAC_marshal");
  if (CL_SUCCESS != kernel.setArg(0, buffer))
    return true;
  cl_int err = kernel.setArg(1, tile_size);
  if (err != CL_SUCCESS)
    return true;
  err |= kernel.setArg(2, (height/tile_size));
  err = kernel.setArg(3, width);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(4, ((tile_size*width+15)/16)*sizeof(cl_uint), NULL);
  if (err != CL_SUCCESS)
    return true;
  cl::NDRange global((height/tile_size+1)/2*NR_THREADS), local(NR_THREADS);
  err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, NULL,
    prof.GetEvent());
  if (err != CL_SUCCESS)
    return true;
#ifdef LIBMARSHAL_OCL_PROFILE
  prof.Report(height*width*sizeof(float)*2);
#endif
  return false;
}

extern "C" bool cl_soa_asta_pttwac(cl_command_queue cl_queue,
    cl_mem src, int height, int width, int tile_size) {
    // Standard preparation of invoking a kernel
  cl::CommandQueue queue = cl::CommandQueue(cl_queue);
  clRetainCommandQueue(cl_queue);
  Profiling prof(queue, "SOA-ASTA PTTWAC");
  cl::Buffer buffer = cl::Buffer(src);
  clRetainMemObject(src);
  cl::Context context;
  if(buffer.getInfo(CL_MEM_CONTEXT, &context) != CL_SUCCESS)
    return true;
  MarshalProg *marshalprog = MarshalProgSingleton::Instance();
  marshalprog->Init(context());

  assert ((height/tile_size)*tile_size == height);
  cl_int *finished = (cl_int *)calloc(sizeof(cl_int),
      height*width/tile_size);
  cl_int err;
  cl::Buffer d_finished = cl::Buffer(context, CL_MEM_READ_WRITE,
      sizeof(cl_int)*height*width/tile_size, NULL, &err);
  if (err != CL_SUCCESS)
    return true;
  err = queue.enqueueWriteBuffer(d_finished, CL_TRUE, 0,
      sizeof(cl_int)*height*width/tile_size, finished);
  if (err != CL_SUCCESS)
    return true;
  cl::Kernel kernel(marshalprog->program, "PTTWAC_marshal_soa");
  if (CL_SUCCESS != kernel.setArg(0, buffer))
    return true;
  err = kernel.setArg(1, height);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(2, tile_size);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(3, width);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(4, d_finished);
  if (err != CL_SUCCESS)
    return true;
  cl::NDRange global(std::min(height*width, tile_size*1024)), local(tile_size);
  err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local,
    NULL, prof.GetEvent());
  if (err != CL_SUCCESS)
    return true;
#ifdef LIBMARSHAL_OCL_PROFILE
  prof.Report(height*width*sizeof(float)*2);
#endif
  return false;
}

extern "C" bool cl_aos_asta(cl_command_queue queue, cl_mem src, int height,
  int width, int tile_size) {
  return cl_aos_asta_bs(queue, src, height, width, tile_size) &&
    cl_aos_asta_pttwac(queue, src, height, width, tile_size);
}
