//===--- cl_aos_astal.cl  - GPU in-place marshaling library    ----------===//
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
//  This file defines the OpenCL kernels of the libmarshal 
//
//===---------------------------------------------------------------------===//


#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

// limitations: tile_size * width cannot exceed maximal # of threads in
// a block allowed in the system
// convert a[height/tile_size][tile_size][width] to
// a[height/tile_size][width][tile_size]
// Launch height/tile_size blocks of tile_size*width threads
__kernel void BS_marshal (__global float *input, int tile_size, int width,
  __local float *store) {
  int tidx = get_local_id(0);
  int m = width*tile_size-1;
  int bid = get_group_id(0);
  input += tile_size*width*bid;
  for (int i = tidx; i < tile_size*width; i+=get_local_size(0)) {
    int next = (i * tile_size)-m*(i/width);
    store[next] = input[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int i = tidx; i < tile_size*width; i+=get_local_size(0)) {
    input[i] = store[i];
  }
}
// Optimized version for tile size == power of 2
// Padding shared memory is necessary to reduce conflicts
__kernel void BS_marshal_power2 (__global float *input, int lg2_tile_size, int width,
  __local float *store) {
#define SHMT lg2_tile_size 
  int tidx = get_local_id(0);
  int m = (width<<SHMT)-1;
  int bid = get_group_id(0);
  input += ((width*bid)<<SHMT); //tile_size*width*bid;
  for (int i = tidx; i < (width<<SHMT) ; i+=get_local_size(0)) {
    int next = (i << SHMT)-m*(i/width);
#define PAD(x) ((x)+((x)>>SHMT))
    store[PAD(next)] = input[i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int i = tidx; i < (width<<SHMT); i+=get_local_size(0)) {
    input[i] = store[PAD(i)];
  }
}

// limitations: height must be multiple of tile_size
// convert a[height/tile_size][tile_size][width] to
// a[height/tile_size][width][tile_size]
// Launch height/tile_size blocks of NR_THREADS threads
__kernel void PTTWAC_marshal(__global float *input, int tile_size, 
  int nr_block, int width, __local uint *finished) {
  int tidx = get_local_id(0);
  int m = tile_size*width - 1;
  int height = nr_block * get_local_size(0);
  __global float *input1 = input + get_group_id(0)*2*tile_size*width;
  __global float *input2 = input + (get_group_id(0)*2+1)*tile_size*width;
  int do_second = true;
  if ((get_group_id(0)*2)>=(nr_block-1))
    do_second = false;
  for (int id = tidx ; id < (tile_size * width + 15) / 16;
      id += get_local_size(0)) {
    finished[id] = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
  for (;tidx < tile_size*width; tidx += get_local_size(0)) {
    int next = (tidx * tile_size)-m*(tidx/width);
    if (tidx != m && next != tidx) {
      float data1 = input1[tidx];
      float data3 = do_second?input2[tidx]:0.0f;
      unsigned int mask = (1 << (tidx % 32));
#define SHFT 4
      unsigned int flag_id = tidx>>SHFT;
      uint done = atom_or(finished+flag_id, 0);
      done = (done & mask);
      for (; done == 0; next = (next * tile_size) - m*(next/width)) {
        
        float data2 = input1[next];
        float data4 = do_second?input2[next]:0.0f;
        mask = (1 << (next % 32));
        flag_id = next>>SHFT;
        done = atom_or(finished+flag_id, mask);
        done = (done & mask);
        if (done == 0) {
          input1[next] = data1;
          if(do_second) input2[next] = data3;
        }
        data1 = data2;
        data3 = data4;
      }
    }
  }
}

// Transformation 100, or ABb to BAb
// limitations: b cannot exceed # of allowed threads in the system
// Launch A*B work-groups of b work-items
__kernel void transpose_100(__global float *input, 
    int A, int B, int b, __global int *finished) {
  int m = A*B-1;
  int tid = get_local_id(0);
  float data;
  for(int gid = get_group_id(0); gid < m; gid += get_num_groups(0)) {
    int next_in_cycle = (gid * A)-m*(gid/B);
    if (next_in_cycle == gid)
      continue;
#define P_IPT 0
#if P_IPT
    for (;next_in_cycle > gid;
      next_in_cycle = (next_in_cycle*A)-m*(next_in_cycle/B))
      ;
    if (next_in_cycle !=gid)
      continue;
    data = input[gid*b+tid];
    for (next_in_cycle = (gid * A)-m*(gid/B);
      next_in_cycle > gid;
      next_in_cycle = (next_in_cycle*A)-m*(next_in_cycle/B)) {
      float backup = input[next_in_cycle*b+tid];
      input[next_in_cycle*b+tid] = data;
      data = backup;
    }
    input[gid*b+tid] = data;
#else
    __local int done;
    data = input[gid*b+tid];
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    if (tid == 0)
      done = atom_or(finished+gid, (int)0); //make sure the read is not cached 
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

    for (;done == 0; next_in_cycle = (next_in_cycle*A)-m*(next_in_cycle/B)) {
      float backup = input[next_in_cycle*b+tid];
      barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
      if (tid == 0) {
        done = atom_xchg(finished+next_in_cycle, (int)1);
      }
      barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
      if (!done) {
        input[next_in_cycle*b+tid] = data;
      }
      data = backup;
    }
#endif
#undef P_IPT
  }
}

// Transformation 0100, or AaBb to ABab
// There are aB workgroups of N*b workitems.
// Each workgroup moves a cycle as in SoA-ASTA transformation
// when moving, N instances of elements of b is moved
// When A == 1 this is equivalent to SoA-ASTA transformation
#define P_IPT 1
__kernel void transpose_0100(
  __global float *input, int A, int a, int B, int b
#if P_IPT
  ) {
#else
  ,__global uint *finished) {
#endif
  int m = (a*B)-1;
  int tid = get_local_id(0);
  float data;
  for(int gid = get_group_id(0); gid < m; gid += get_num_groups(0)) {
    int next_in_cycle = (gid * a)-m*(gid/B);
    if (next_in_cycle == gid)
      continue;
#if P_IPT
    for (;next_in_cycle > gid;
      next_in_cycle = (next_in_cycle * a)-m*(next_in_cycle/B))
      ;
    if (next_in_cycle !=gid)
      continue;
    for (int i = get_local_id(1); i < A; i += get_local_size(1)) {
      data = input[i*a*B*b+gid*b+tid];
      for (next_in_cycle = (gid * a)-m*(gid/B);
          next_in_cycle > gid;
          next_in_cycle = (next_in_cycle*a)-m*(next_in_cycle/B)) {
      float backup = input[i*a*B*b+next_in_cycle*b+tid];
      input[i*a*B*b+next_in_cycle*b+tid] = data;
      data = backup;
      }
      input[i*a*B*b+gid*b+tid] = data;
    }
#else
    // Works only when A*b < Max # of work-items allowed in a workgroup
    __local int done;
    data = input[get_local_id(1)*a*B*b+gid*b+tid];
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    if (get_local_id(1) == 0 && tid == 0)
      done = atom_or(finished+gid, (int)0); //make sure the read is not cached 
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

    for (;done == 0; next_in_cycle = (next_in_cycle*a)-m*(next_in_cycle/B)) {
      float backup;
      backup = input[get_local_id(1)*a*B*b+next_in_cycle*b+tid];
      barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
      if (get_local_id(1) == 0 && tid == 0) {
        done = atom_xchg(finished+next_in_cycle, (int)1);
      }
      barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
      if (!done) {
        input[get_local_id(1)*a*B*b + next_in_cycle*b+tid] = data;
      }
      data = backup;
    }
#endif
  }
}

