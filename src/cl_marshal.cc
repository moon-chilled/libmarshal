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
  void Finalize(void) {
    program = cl::Program();
    context_ = NULL;
  }
  cl::Program program;
 private:
  cl::Program::Sources source_;
  std::string source_code_;
  cl_context context_;
};
typedef Singleton<MarshalProg> MarshalProgSingleton;
}
extern "C" void cl_marshal_finalize(void) {
  MarshalProg *marshalprog = MarshalProgSingleton::Instance();
  marshalprog->Finalize();
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

// Transformation 010, or AaB to ABa
extern "C" bool cl_transpose_010_bs(cl_command_queue cl_queue,
    cl_mem src, int A, int a,
    int B) {
  // Standard preparation of invoking a kernel
  cl::CommandQueue queue = cl::CommandQueue(cl_queue);
  Profiling prof(queue, "AOS-ASTA BS");
  cl::Buffer buffer = cl::Buffer(src);
  clRetainMemObject(src);
  cl::Context context;
  if(buffer.getInfo(CL_MEM_CONTEXT, &context) != CL_SUCCESS)
    return true;
  MarshalProg *marshalprog = MarshalProgSingleton::Instance();
  marshalprog->Init(context());

  cl::Kernel kernel(marshalprog->program, 
      IS_POW2(a) ? "BS_marshal_power2":"BS_marshal");
  if (CL_SUCCESS != kernel.setArg(0, buffer))
    return true;
  cl_int err = kernel.setArg(1,
      IS_POW2(a)? count_zero_bits(a) : a);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(2, B);
  err |= kernel.setArg(3, B*(a+1)*sizeof(cl_float), NULL);
  if (err != CL_SUCCESS)
    return true;
  cl::NDRange global(A*NR_THREADS), local(NR_THREADS);
  err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, NULL,
    prof.GetEvent());
  if (err != CL_SUCCESS)
    return true;
#ifdef LIBMARSHAL_OCL_PROFILE
  prof.Report(A*a*B*sizeof(float)*2);
#endif
  return false;
}

// Transformation 010, or AaB to ABa
extern "C" bool cl_transpose_010_pttwac(cl_command_queue cl_queue,
    cl_mem src, int A, int a, int B) {
  // Standard preparation of invoking a kernel
  cl::CommandQueue queue = cl::CommandQueue(cl_queue);
  Profiling prof(queue, "AOS-ASTA PTTWAC");
  cl::Buffer buffer = cl::Buffer(src);
  clRetainMemObject(src);
  cl::Context context;
  if(buffer.getInfo(CL_MEM_CONTEXT, &context) != CL_SUCCESS)
    return true;
  MarshalProg *marshalprog = MarshalProgSingleton::Instance();
  marshalprog->Init(context());

  cl::Kernel kernel(marshalprog->program, "PTTWAC_marshal");
  if (CL_SUCCESS != kernel.setArg(0, buffer))
    return true;
  cl_int err = kernel.setArg(1, a);
  if (err != CL_SUCCESS)
    return true;
  err |= kernel.setArg(2, A);
  err = kernel.setArg(3, B);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(4, ((a*B+15)/16)*sizeof(cl_uint), NULL);
  if (err != CL_SUCCESS)
    return true;
  cl::NDRange global((A+1)/2*NR_THREADS), local(NR_THREADS);
  err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, NULL,
    prof.GetEvent());
  if (err != CL_SUCCESS)
    return true;
#ifdef LIBMARSHAL_OCL_PROFILE
  prof.Report(A*a*B*sizeof(float)*2);
#endif
  return false;
}

// Transformation 100, or ABb to BAb
extern "C" bool cl_transpose_100(cl_command_queue cl_queue,
    cl_mem src, int A, int B, int b) {
    // Standard preparation of invoking a kernel
  cl::CommandQueue queue = cl::CommandQueue(cl_queue);
  Profiling prof(queue, "SOA-ASTA PTTWAC");
  cl::Buffer buffer = cl::Buffer(src);
  clRetainMemObject(src);
  cl::Context context;
  if(buffer.getInfo(CL_MEM_CONTEXT, &context) != CL_SUCCESS)
    return true;
  MarshalProg *marshalprog = MarshalProgSingleton::Instance();
  marshalprog->Init(context());

  cl_int *finished = (cl_int *)calloc(sizeof(cl_int), A*B);
  cl_int err;
  cl::Buffer d_finished = cl::Buffer(context, CL_MEM_READ_WRITE,
      sizeof(cl_int)*A*B, NULL, &err);
  if (err != CL_SUCCESS)
    return true;
  err = queue.enqueueWriteBuffer(d_finished, CL_TRUE, 0,
      sizeof(cl_int)*A*B, finished);
  free(finished);
  if (err != CL_SUCCESS)
    return true;
  cl::Kernel kernel(marshalprog->program, "transpose_100");
  if (CL_SUCCESS != kernel.setArg(0, buffer))
    return true;
  err = kernel.setArg(1, A);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(2, B);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(3, b);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(4, d_finished);
  if (err != CL_SUCCESS)
    return true;
  cl::NDRange global(std::min(A*B*b, b*1024)), local(b);
  err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local,
    NULL, prof.GetEvent());
  if (err != CL_SUCCESS)
    return true;
#ifdef LIBMARSHAL_OCL_PROFILE
  prof.Report(A*B*b*sizeof(float)*2);
#endif
  return false;
}

extern "C" bool cl_transpose(cl_command_queue queue, cl_mem src, int height,
  int width, int tile_size) {
  std::cerr << "cl_transpose: tile size = " << tile_size << "\n";
  // Method 1: H >> W
  // [H/T][T][W] to [H/T][W][T] then
  // [H/T][W][T] to [W][H/T][T]
  return cl_aos_asta(queue, src, height, width, tile_size) ||
    cl_soa_asta_pttwac(queue, src, width*tile_size, height/tile_size, tile_size);
  // Method 2: W >> H (TBD)
  // Method 3: 
}

// Transformation 0100, or AaBb to ABab
extern "C" bool cl_transpose_0100(cl_command_queue cl_queue, cl_mem src,
  int A, int a, int B, int b) {
  // Standard preparation of invoking a kernel
  cl::CommandQueue queue = cl::CommandQueue(cl_queue);
  Profiling prof(queue, "Transpostion 0100 (PTTWAC)");
  cl::Buffer buffer = cl::Buffer(src);
  clRetainMemObject(src);
  cl::Context context;
  if(buffer.getInfo(CL_MEM_CONTEXT, &context) != CL_SUCCESS)
    return true;
  MarshalProg *marshalprog = MarshalProgSingleton::Instance();
  marshalprog->Init(context());

  cl::Kernel kernel(marshalprog->program, "transpose_0100");
  if (CL_SUCCESS != kernel.setArg(0, buffer))
    return true;
  cl_int err = kernel.setArg(1, A);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(2, a);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(3, B);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(4, b);
  if (err != CL_SUCCESS)
    return true;
#if 0 //PTTWAC
  cl::Buffer d_finished = cl::Buffer(context, CL_MEM_READ_WRITE,
      sizeof(cl_int)*a*B, NULL, &err);
  {
    cl_int *finished = (cl_int *)calloc(sizeof(cl_int),
        a*B);
    cl_int err;
    if (err != CL_SUCCESS)
      return true;
    err = queue.enqueueWriteBuffer(d_finished, CL_TRUE, 0,
        sizeof(cl_int)*a*B, finished);
    free(finished);
  }

  err = kernel.setArg(5, d_finished);
  if (err != CL_SUCCESS)
    return true;
  cl::NDRange global(a*B*b, A), local(b, A);
#else //PIPT
  cl::NDRange global(a*B*b, 1024/b), local(b, 1024/b);
#endif
  err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, NULL,
    prof.GetEvent());
  if (err != CL_SUCCESS)
    return true;
#ifdef LIBMARSHAL_OCL_PROFILE
  prof.Report(A*a*B*b*sizeof(float)*2);
#endif
  return false;
}


// Wrappers for old API compatibility
// Transformation 010, or AaB to ABa
extern "C" bool cl_aos_asta_bs(cl_command_queue cl_queue,
    cl_mem src, int height, int width,
    int tile_size) {
  return cl_transpose_010_bs(cl_queue, src,
    height/tile_size /*A*/,
    tile_size /*a*/,
    width /*B*/);
}

extern "C" bool cl_soa_asta_pttwac(cl_command_queue cl_queue,
    cl_mem src, int height, int width, int tile_size) {
  assert ((height/tile_size)*tile_size == height);
  return cl_transpose_100(cl_queue, src, width /*A*/,
    height/tile_size /*B*/, tile_size/*b*/);
}
// Transformation 010, or AaB to ABa
extern "C" bool cl_aos_asta_pttwac(cl_command_queue cl_queue,
    cl_mem src, int height, int width, int tile_size) {
  assert ((height/tile_size)*tile_size == height);
  return cl_transpose_010_pttwac(cl_queue, src, height/tile_size/*A*/, 
    tile_size /*a*/,
    width /*B*/);
}

extern "C" bool cl_aos_asta(cl_command_queue queue, cl_mem src, int height,
  int width, int tile_size) {
  return cl_aos_asta_bs(queue, src, height, width, tile_size) &&
    cl_aos_asta_pttwac(queue, src, height, width, tile_size);
}

