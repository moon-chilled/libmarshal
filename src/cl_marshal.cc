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
#include "plan.hpp"
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
    cl_mem src, int A, int a, int B, cl_ulong *elapsed_time) {
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
  err = kernel.setArg(4, ((a*B+31)/32)*sizeof(cl_uint), NULL);
  if (err != CL_SUCCESS)
    return true;
  cl::NDRange global(A*NR_THREADS), local(NR_THREADS);
  err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, NULL,
    prof.GetEvent());
  if (err != CL_SUCCESS)
    return true;
#ifdef LIBMARSHAL_OCL_PROFILE
  if (elapsed_time) {
    *elapsed_time += prof.Report();
  }
  // prof.Report(A*a*B*sizeof(float)*2);
#endif
  return false;
}

// Transformation 100, or ABb to BAb
extern "C" bool cl_transpose_100(cl_command_queue cl_queue,
    cl_mem src, int A, int B, int b, cl_ulong *elapsed_time) {
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

  // Shared memory tiling
#define WARPS 6
#define WARP_SIZE 32
  err = kernel.setArg(5, b*WARPS*sizeof(cl_float), NULL);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(6, b*WARPS*sizeof(cl_float), NULL);
  if (err != CL_SUCCESS)
    return true;

  //cl::NDRange global(std::min(A*B*b, b*1024)), local(b);
  // Shared memory tiling
  cl::NDRange global(std::min(A*B*WARP_SIZE*WARPS, 1024*WARP_SIZE*WARPS)),
    local(WARP_SIZE*WARPS);

  err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local,
    NULL, prof.GetEvent());
  if (err != CL_SUCCESS)
    return true;
#ifdef LIBMARSHAL_OCL_PROFILE
  if (elapsed_time) {
    *elapsed_time += prof.Report();
  }
  // prof.Report(A*B*b*sizeof(float)*2);
#endif
  return false;
}

extern "C" bool cl_transpose(cl_command_queue queue, cl_mem src, int A, int a,
  int B, int b) {
  cl::Buffer buffer = cl::Buffer(src);
  clRetainMemObject(src);
  cl::Context context;
  if(buffer.getInfo(CL_MEM_CONTEXT, &context) != CL_SUCCESS)
    return true;
  {
    // Method 1: Aa >> Bb
    T010_BS step1(a, B*b, context()); // Aa(Bb) to A(Bb)a
    T010_PTTWAC step1p(a, B*b, context()); // Aa(Bb) to A(Bb)a
    T0100_PTTWAC step2(1, A, B*b, a, context()); //1A(Bb)a to 1(Bb)Aa
    if ((step1.IsFeasible()||step1p.IsFeasible()) &&
         step2.IsFeasible()) {
#ifdef LIBMARSHAL_OCL_PROFILE
      std::cerr << "cl_transpose: method 1\n";
#endif
      bool r1;
      if (step1.IsFeasible())
        r1 = cl_transpose_010_bs(queue, src, A, a, B*b);
      else
        r1 = cl_transpose_010_pttwac(queue, src, A, a, B*b, NULL);
      if (r1)
        std::cerr << "cl_transpose: step 1 failed\n";
      return r1 || cl_transpose_100(queue, src, A, B*b, a, NULL);
    }
  }
  {
    // Method 2: a < MAX_THREADS 
    // AaBb to BAab (step 1)
    // to BAba (step 2)
    // to BbAa (step 3)
    T0100_PTTWAC step1(1, A*a, B, b, context());
    T010_PTTWAC step2(a, b, context());
    T0100_PTTWAC step3(B, A, b, a, context());
    if (step1.IsFeasible() && step2.IsFeasible() && step3.IsFeasible()) {
#ifdef LIBMARSHAL_OCL_PROFILE
      std::cerr << "cl_transpose: method 2\n";
#endif
      return cl_transpose_100(queue, src, A*a, B, b, NULL) ||
        cl_transpose_010_pttwac(queue, src, B*A, a, b, NULL) ||
        cl_transpose_0100(queue, src, B, A, b, a);
    }
  }
  // fallback
#ifdef LIBMARSHAL_OCL_PROFILE
  std::cerr << "cl_transpose: fallback\n";
#endif
  return cl_transpose_0100(queue, src, 1, A*a, B*b, 1);
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
  GPUInfo info(context());
  cl::NDRange global(a*B*b, info.GetMaxWorkItems()/b),
                          local(b, info.GetMaxWorkItems()/b);
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

// Transformation 010, or AaB to ABa
extern "C" bool cl_aos_asta_pttwac(cl_command_queue cl_queue,
    cl_mem src, int height, int width, int tile_size) {
  assert ((height/tile_size)*tile_size == height);
  return cl_transpose_010_pttwac(cl_queue, src, height/tile_size/*A*/, 
    tile_size /*a*/,
    width /*B*/,
    NULL);
}

extern "C" bool cl_aos_asta(cl_command_queue queue, cl_mem src, int height,
  int width, int tile_size) {
  return cl_aos_asta_bs(queue, src, height, width, tile_size) &&
    cl_aos_asta_pttwac(queue, src, height, width, tile_size);
}

