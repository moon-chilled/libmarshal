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
#include <math.h>
#include <stdio.h>

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
#define NR_THREADS 1024
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
    cl_mem src, int A, int a, int B, cl_ulong *elapsed_time) {
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

  int sh_sz = a*B;
  cl::Kernel kernel(marshalprog->program,
      sh_sz<224 ? "BS_marshal_vw" : (IS_POW2(a) ? "BS_marshal_power2":"BS_marshal"));
      //sh_sz<224 ? "BS_marshal_vw" : "BS_marshal");
  if (CL_SUCCESS != kernel.setArg(0, buffer))
    return true;
  cl_int err = kernel.setArg(1,
      sh_sz<224 ? a: (IS_POW2(a)? count_zero_bits(a) : a));
      //a);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(2, B);
  if (err != CL_SUCCESS)
    return true;

  // Select work-group size and virtual-SIMD-unit size
  int nr_threads;
  int warp_size;
  // Block size (if block-centric)
  if (sh_sz <= 768) nr_threads = 128;
  else if (sh_sz > 768 && sh_sz <= 1792) nr_threads = 256;
  else if (sh_sz > 1792 && sh_sz <= 3072) nr_threads = 512;
  else nr_threads = 1024;
  // Virtual warps
  if (sh_sz < 4) warp_size = 1;
  else if (sh_sz >= 4 && sh_sz < 8) warp_size = 2;
  else if (sh_sz >= 8 && sh_sz < 16) warp_size = 4;
  else if (sh_sz >= 16 && sh_sz < 32) warp_size = 8;
  else if (sh_sz >= 32 && sh_sz < 96) warp_size = 16;
  else warp_size = 32;

  int warps = nr_threads / warp_size;
  //err = kernel.setArg(3, sh_sz<224?(warps*B*(a+1)*sizeof(cl_float)):(B*(a+1)*sizeof(cl_float)), NULL);
  err = kernel.setArg(3, sh_sz<224?(warps*B*a*sizeof(cl_float)):(IS_POW2(a)?(B*(a+1)*sizeof(cl_float)):B*a*sizeof(cl_float)), NULL);
  err |= kernel.setArg(4, warp_size);
  err |= kernel.setArg(5, A);
  if (err != CL_SUCCESS)
    return true;

  // NDRange and kernel call
  if (sh_sz < 224){
    std::cerr << "nr_threads_vwarp = " << warp_size << "\t"; // Print warp_size
    //cl::NDRange global((A/warps+1)*nr_threads), local(nr_threads);
    cl::NDRange global(std::min((A/warps+1)*nr_threads, (8192/warps+1)*nr_threads)), local(nr_threads);
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, NULL,
      prof.GetEvent());
    if (err != CL_SUCCESS)
      return true;
  }
  else{
    std::cerr << "nr_threads =  " << nr_threads << "\t"; // Print nr_threads
    //cl::NDRange global(A*nr_threads), local(nr_threads);
    cl::NDRange global(std::min(A*nr_threads, 8192*nr_threads)), local(nr_threads);
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, NULL,
      prof.GetEvent());
    if (err != CL_SUCCESS)
      return true;
  }

#ifdef LIBMARSHAL_OCL_PROFILE
  if (elapsed_time) {
    *elapsed_time += prof.Report();
  } else {
    prof.Report(A*a*B*sizeof(float)*2);
  }
#endif
  return false;
}

// Transformation 010, or AaB to ABa
extern "C" bool cl_transpose_010_pttwac(cl_command_queue cl_queue,
    cl_mem src, int A, int a, int B, cl_ulong *elapsed_time, int R, int P) {
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

  cl::Kernel kernel(marshalprog->program, "transpose_010_PTTWAC");
  if (CL_SUCCESS != kernel.setArg(0, buffer))
    return true;
  cl_int err = kernel.setArg(1, A);
  if (err != CL_SUCCESS)
    return true;
  err |= kernel.setArg(2, a);
  err = kernel.setArg(3, B);
  if (err != CL_SUCCESS)
    return true;

  int sh_sz = R * ((a*B+31)/32);
  sh_sz += (sh_sz >> 5) * P;
  err = kernel.setArg(4, sh_sz*sizeof(cl_uint), NULL);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(5, R);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(6, (int)(5 - log2(R)));
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(7, P);
  if (err != CL_SUCCESS)
    return true;

  // NDRange and kernel call
  std::cerr << "NR_THREADS = " << NR_THREADS << "\t"; // Print nr_threads
  cl::NDRange global(A*NR_THREADS), local(NR_THREADS);
  err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, NULL,
    prof.GetEvent());
  if (err != CL_SUCCESS)
    return true;
#ifdef LIBMARSHAL_OCL_PROFILE
  if (elapsed_time) {
    *elapsed_time += prof.Report();
  } else {
    prof.Report(A*a*B*sizeof(float)*2);
  }
#endif
  return false;
}

// Generic transformation 0100, or AaBb to ABab. Used by both
// transformation 100 and 0100.
bool _cl_transpose_0100(cl_command_queue cl_queue,
    cl_mem src, int A, int a, int B, int b, cl_ulong *elapsed_time) {
    // Standard preparation of invoking a kernel
  cl::CommandQueue queue = cl::CommandQueue(cl_queue);
  Profiling prof(queue, "Transpose 0100/100 PTTWAC");
  cl::Buffer buffer = cl::Buffer(src);
  clRetainMemObject(src);
  cl::Context context;
  if(buffer.getInfo(CL_MEM_CONTEXT, &context) != CL_SUCCESS)
    return true;
  MarshalProg *marshalprog = MarshalProgSingleton::Instance();
  marshalprog->Init(context());

  cl_int *finished = (cl_int *)calloc(sizeof(cl_int), A*a*B);
  cl_int err;
  cl::Buffer d_finished = cl::Buffer(context, CL_MEM_READ_WRITE,
      sizeof(cl_int)*A*a*B, NULL, &err);
  if (err != CL_SUCCESS)
    return true;
  err = queue.enqueueWriteBuffer(d_finished, CL_TRUE, 0,
      sizeof(cl_int)*A*a*B, finished);
  free(finished);
  if (err != CL_SUCCESS)
    return true;

  cl::Kernel kernel(marshalprog->program, 
    b<192?(A==1?"transpose_100":"transpose_0100"):(A==1?"transpose_100_b":"transpose_0100_b"));

  err = kernel.setArg(0, buffer);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(1, a);
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

  // Select work-group size and virtual-SIMD-unit size
  // This version uses shared memory tiling
#define WARPS 4
#define WARP_SIZE 32
  // Virtual warps
  int v_warp_size;
  if (b <= 1) v_warp_size = 1;
  else if (b > 1 && b <= 2) v_warp_size = 2;
  else if (b > 2 && b <= 4) v_warp_size = 4;
  else if (b > 4 && b <= 24) v_warp_size = 8;
  else if (b > 24 && b <= 48) v_warp_size = 16;
  else v_warp_size = 32;
  // Block size (if block-centric)
  int block_size;
  if (b <= 512) block_size = 128;
  else if (b > 512 && b <= 1024) block_size = 256;
  else if (b > 1024 && b <= 1536) block_size = 512;
  else block_size = 1024;

  err = kernel.setArg(5, b<192?(b*WARPS*(WARP_SIZE/v_warp_size)*sizeof(cl_float)):(b*sizeof(cl_float)), NULL);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(6, b<192?(b*WARPS*(WARP_SIZE/v_warp_size)*sizeof(cl_float)):(b*sizeof(cl_float)), NULL);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(7, v_warp_size);
  if (err != CL_SUCCESS)
    return true;
  err = kernel.setArg(8, b<192?(WARPS*(WARP_SIZE/v_warp_size)*sizeof(cl_int)):(sizeof(cl_int)), NULL);
  if (err != CL_SUCCESS)
    return true;

  // NDRange and kernel call
  if (b < 192){
    std::cerr << "vwarp = " << v_warp_size << "\t"; // Print v_warp_size
    // NDRange - PPoPP'2014 + use of virtual warps
    cl::NDRange global(std::min(a*B*WARP_SIZE, 1024*WARP_SIZE), WARPS, A),
      local(WARP_SIZE, WARPS, 1);
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local,
      NULL, prof.GetEvent());
    if (err != CL_SUCCESS)
      return true;
  }
  else{
    std::cerr << "blocksize =  " << block_size << "\t"; // Print block size
    // NDRange - Block-centric using shared memory tiling
    cl::NDRange global(std::min(a*B*block_size, 1024*block_size), 1, A),
      local(block_size, 1, 1);
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local,
      NULL, prof.GetEvent());
    if (err != CL_SUCCESS)
      return true;
  }

#ifdef LIBMARSHAL_OCL_PROFILE
  if (elapsed_time) {
    *elapsed_time += prof.Report();
  } else {
    prof.Report(A*a*B*b*sizeof(float)*2);
  }
#endif
  return false;
}

extern "C" bool cl_transpose(cl_command_queue queue, cl_mem src, int A, int a,
  int B, int b, int R, cl_ulong *elapsed_time) {
  cl::Buffer buffer = cl::Buffer(src);
  clRetainMemObject(src);
  cl::Context context;
  if(buffer.getInfo(CL_MEM_CONTEXT, &context) != CL_SUCCESS)
    return true;
  cl_ulong et = 0;
  if (0){
    // Method 1: Aa >> Bb
    //T010_BS step1(a, B*b, context()); // Aa(Bb) to A(Bb)a
    //T010_PTTWAC step1p(a, B*b, context()); // Aa(Bb) to A(Bb)a
    //T0100_PTTWAC step2(1, A, B*b, a, context()); //1A(Bb)a to 1(Bb)Aa
    //if ((step1.IsFeasible()||step1p.IsFeasible()) &&
    //     step2.IsFeasible()) {
    if (((a*B*b+31)/32) + ((((a*B*b+31)/32)>>5)*1) <= 12288 && a < (12288 - 64)/2){
      bool r1;
      //if (step1.IsFeasible())
      if ((IS_POW2(a) && B*b*(a+1) <= 12288) || (!IS_POW2(a) && B*b*a <= 12288))
        r1 = cl_transpose_010_bs(queue, src, A, a, B*b, &et);
      else
        r1 = cl_transpose_010_pttwac(queue, src, A, a, B*b, &et, R, 1);
      if (r1) {
        std::cerr << "cl_transpose: step 1 failed\n";
        return r1;
      }
      bool r2 = cl_transpose_100(queue, src, A, B*b, a, &et);
      if (r2) {
        std::cerr << "cl_transpose: step 2 failed\n";
      }
#ifdef LIBMARSHAL_OCL_PROFILE
      //std::cerr << "[cl_transpose] method 1; "<< 
      std::cerr << 
        float(A*a*B*b*2*sizeof(float))/et << "\n";
#endif
      return r2;
    }
  }
  // 3-step approach
  if (1) {
    // Method 2: a, b < TILE_SIZE 
    // AaBb to BAab (step 1)
    // to BAba (step 2)
    // to BbAa (step 3)
    //T0100_PTTWAC step1(1, A*a, B, b, context());
    //T010_BS step2(a, b, context());
    //T010_PTTWAC step2p(a, b, context());
    //T0100_PTTWAC step3(B, A, b, a, context());
    //if (step1.IsFeasible() && (step2.IsFeasible()||step2p.IsFeasible()) 
    //    && step3.IsFeasible()) {
    //if (((a*b+31)/32) + ((((a*b+31)/32)>>5)*P) <= 12288 - 512 && b < (12288 - 512)/2 && a < (12288 - 512)/2){
    //if (((a*b+31)/32) + ((((a*b+31)/32)>>5)*1) <= 12288 && b < (12288 - 64)/2 && a < (12288 - 64)/2 && B < 2000){
    if (((a*b+31)/32) + ((((a*b+31)/32)>>5)*1) <= 12288 && b < (12288 - 64)/2 && a < (12288 - 64)/2){
      bool r1 = cl_transpose_100(queue, src, A*a, B, b, &et);
      if (r1) {
        std::cerr << "cl_transpose: step 2.1 failed\n";
        return r1;
      }
      bool r2;
      //if (step2.IsFeasible()){
      if ((IS_POW2(a) && b*(a+1) <= 12288) || (!IS_POW2(a) && b*a <= 12288)){
        std::cerr << "010_BS\t";
        r2 = cl_transpose_010_bs(queue, src, B*A, a, b, &et);
      }
      else{
        std::cerr << "010_PTTWAC\t";
        r2 = cl_transpose_010_pttwac(queue, src, B*A, a, b, &et, R, 1);
      }
      if (r2) {
        std::cerr << "cl_transpose: step 2.2 failed\n";
        return r2;
      }
      bool r3 = cl_transpose_0100(queue, src, B, A, b, a, &et);
      if (r3) {
        std::cerr << "cl_transpose: step 2.3 failed\n";
        return r3;
      }
#ifdef LIBMARSHAL_OCL_PROFILE
      //std::cerr << "[cl_transpose] method 2; "<< 
      std::cerr<<
        float(A*a*B*b*2*sizeof(float))/et << "\n";
#endif
      return r1 || r2 || r3;
    }
  }
  // 4-step approach
  if (0) {
    // Karlsson's method: a, b < TILE_SIZE 
    // AaBb to ABab (0100)
    // ABab ABba    (0010)
    // ABba BAba    (1000)
    // BAba to BbAa (0100)
    T0100_PTTWAC step1(A, a, B, b, context());
    T010_BS step2(a, b, context());
    T010_PTTWAC step2p(a, b, context());
    T0100_PTTWAC step3(1, A, B, b*a, context());
    T0100_PTTWAC step4(B, A, b, a, context());
    if (step1.IsFeasible() && (step2.IsFeasible()||step2p.IsFeasible()) 
        && step3.IsFeasible() && step4.IsFeasible()) {
      bool r1 = cl_transpose_0100(queue, src, A, a, B, b, &et);
      if (r1) {
        std::cerr << "cl_transpose: step 4.1 failed\n";
        return r1;
      }
      bool r2;
      if (step2.IsFeasible())
        r2 = cl_transpose_010_bs(queue, src, B*A, a, b, &et);
      else
        r2 = cl_transpose_010_pttwac(queue, src, B*A, a, b, &et, R, 1);
      if (r2) {
        std::cerr << "cl_transpose: step 4.2 failed\n";
        return r2;
      }
      bool r3 = cl_transpose_100(queue, src, A, B, b*a, &et);
      if (r3) {
        std::cerr << "cl_transpose: step 4.3 failed\n";
        return r3;
      }
      bool r4 = cl_transpose_0100(queue, src, B, A, b, a, &et);
      if (r4) {
        std::cerr << "cl_transpose: step 4.4 failed\n";
        return r4;
      }
#ifdef LIBMARSHAL_OCL_PROFILE
      //std::cerr << "[cl_transpose] Karlsson's method; "<< 
      std::cerr<<
        float(A*a*B*b*2*sizeof(float))/et << "\n";
#endif
      return r1 || r2 || r3 || r4;
    }
  }

  // fallback
bool r;
if(A*a<8000 && B*b<8000)
  //bool r = cl_transpose_0100(queue, src, 1, A*a, B*b, 1, &et);
  r = cl_transpose_0100(queue, src, 1, A*a, B*b, 1, &et);
#ifdef LIBMARSHAL_OCL_PROFILE
  //std::cerr << "[cl_transpose] fallback; "<< 
  std::cerr<<
    float(A*a*B*b*2*sizeof(float))/et << "\n";
#endif
  return r;
}

// Transformation 100, or ABb to BAb
extern "C" bool cl_transpose_100(cl_command_queue cl_queue,
    cl_mem src, int A, int B, int b, cl_ulong *elapsed_time) {
  return _cl_transpose_0100(cl_queue, src, 1, A, B, b, elapsed_time);
}

// Transformation 0100, or AaBb to ABab
extern "C" bool cl_transpose_0100(cl_command_queue cl_queue, cl_mem src,
  int A, int a, int B, int b, cl_ulong *elapsed_time) {
  return _cl_transpose_0100(cl_queue, src, A, a, B, b, elapsed_time);
}


// Wrappers for old API compatibility
// Transformation 010, or AaB to ABa
extern "C" bool cl_aos_asta_bs(cl_command_queue cl_queue,
    cl_mem src, int height, int width,
    int tile_size) {
  return cl_transpose_010_bs(cl_queue, src,
    height/tile_size /*A*/,
    tile_size /*a*/,
    width /*B*/, NULL);
}

// Transformation 010, or AaB to ABa
extern "C" bool cl_aos_asta_pttwac(cl_command_queue cl_queue,
    cl_mem src, int height, int width, int tile_size, int R, int P) {
  assert ((height/tile_size)*tile_size == height);
  return cl_transpose_010_pttwac(cl_queue, src, height/tile_size/*A*/, 
    tile_size /*a*/,
    width /*B*/,
    NULL,
    R,
    P);
}

extern "C" bool cl_aos_asta(cl_command_queue queue, cl_mem src, int height,
  int width, int tile_size, int R) {
  return cl_aos_asta_bs(queue, src, height, width, tile_size) &&
    cl_aos_asta_pttwac(queue, src, height, width, tile_size, R, 1);
}

