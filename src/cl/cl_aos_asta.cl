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
__kernel void BS_marshal (__global float *input, int tile_size, int width) {
  int tidx = get_local_id(1);
  int tidy = get_local_id(0);
  int bid = get_group_id(0);
  input += tile_size*width*bid;
  float tmp = input[tidy*width+tidx];
  barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
  input[tidx*tile_size+tidy] = tmp;
}

// limitations: height must be multiple of tile_size
// convert a[height/tile_size][tile_size][width] to
// a[height/tile_size][width][tile_size]
// Launch height/tile_size blocks of NR_THREADS threads
__kernel void PTTWAC_marshal(__global float *input, int tile_size, int width,
    __local uint *finished) {
  int tidx = get_local_id(0);
  int m = tile_size*width - 1;
  input += get_group_id(0)*tile_size*width;
  for (int id = tidx ; id < (tile_size * width + 31) / 32;
      id += get_local_size(0)) {
    finished[id] = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
  for (;tidx < tile_size*width; tidx += get_local_size(0)) {
    int next = (tidx * tile_size) % m;
    if (tidx != m && next != tidx) {
      float data1 = input[tidx];
      unsigned int mask = (1 << (tidx % 32));
      unsigned int flag_id = (((unsigned int) tidx) >> 5);
      int done = atom_or(finished+flag_id, 0);
      done = (done & mask);
      for (; done == 0; next = (next * tile_size) % m) {
        float data2 = input[next];
        mask = (1 << (next % 32));
        flag_id = (((unsigned int)next) >> 5);
        done = atom_or(finished+flag_id, mask);
        done = (done & mask);
        if (done == 0) {
          input[next] = data1;
        }
        data1 = data2;
      }
    }
  }
}

// limitations: tile_size cannot exceed # of allowed threads in the system
// convert a[width][height/tile_size][tile_size] to
// a[height/tile_size][width][tile_size]
// Launch width*height/tile_size blocks of tile_size threads
__kernel void PTTWAC_marshal_soa(__global float *input, 
    int height, int tile_size,
    int width, __global int *finished) {
  int m = (height*width)/tile_size-1;
  int tid = get_local_id(0);
  float data;
  for(int gid = get_group_id(0); gid < m; gid += get_num_groups(0)) {
    int next_in_cycle = (gid * width)%m;
    if (next_in_cycle == gid)
      continue;

    __local int done;
    data = input[gid*tile_size+tid];
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    if (tid == 0)
      done = atom_or(finished+gid, (int)0); //make sure the read is not cached 
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

    for (;done == 0; next_in_cycle = (next_in_cycle*width)%m) {
      float backup = input[next_in_cycle*tile_size+tid];
      barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
      if (tid == 0) {
        done = atom_xchg(finished+next_in_cycle, (int)1);
      }
      barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
      if (!done) {
        input[next_in_cycle*tile_size+tid] = data;
      }
      data = backup;
    }
  }
}

