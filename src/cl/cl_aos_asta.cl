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
// Contributor: Juan Gómez Luna (el1goluj@uco.es)
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


__kernel void mymemset (__global float *input) {
  input[get_global_id(0)] = 0.0f;
}

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

// Converts input[A][a][B] to input[A][B][a]
// Launch A blocks of NR_THREADS threads
__kernel void transpose_010_PTTWAC(__global float *input, int A, 
  int a, int B, __local uint *finished, int R, int SHFT) {
  int tidx = get_local_id(0);
  int m = a*B - 1;
  input += get_group_id(0)*a*B;

#define P 1
  int sh_sz = R * ((a * B + 31) / 32);
  sh_sz += (sh_sz >> 5) * P; // Padding each 32 locations (Number of banks)


#define P_IPT 0
#define PTTWAC_REMAP 0
#if !P_IPT
  //for (int id = tidx ; id < (tile_size * width + 31) / 32;
  for (int id = tidx ; id < sh_sz;
      id += get_local_size(0)) {
    finished[id] = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
#endif
  for (;tidx < m; tidx += get_local_size(0)) {
    int next = (mul24(tidx, a))-m*(tidx/B);
#if P_IPT
    int prev = (mul24(tidx, B))-m*(tidx/a);
    if (next <= tidx || prev < tidx)
      continue;
    for (;next > tidx;next = (mul24(next, a))-m*(next/B))
      ;
    if (next !=tidx)
      continue;
    float data = input[tidx];
    for (next = (tidx * a)-m*(tidx/B);next > tidx;
        next = (next * a)-m*(next/B)) {
      float backup = input[next];
      input[next] = data;
      data = backup;
    }
    input[tidx] = data;
#else // P_IPT
    if (next != tidx) {
      float data1 = input[tidx];

      //// mask and flag_id////
#if PTTWAC_REMAP
      unsigned int mask = 1 << (tidx / sh_sz);
      unsigned int flag_id = tidx%sh_sz;
#else
      unsigned int mask = (1 << (tidx % 32));
      unsigned int flag_id = tidx >> SHFT;
      flag_id += (flag_id >> 5) * P;
#endif

      uint done = atom_or(finished+flag_id, 0);
      done = (done & mask);

      for (; done == 0; next = (next * a) - m*(next/B)) {
        float data2 = input[next];
#if PTTWAC_REMAP
        mask = 1 << (next / sh_sz);
        flag_id = next%sh_sz;
#else
        //// mask and flag_id////
        mask = (1 << (next % 32));
        flag_id = next >> SHFT;
        flag_id += (flag_id >> 5) * P;
#endif

        done = atom_or(finished+flag_id, mask);
        done = (done & mask);
        if (done == 0) {
          input[next] = data1;
        }
        data1 = data2;
      }
    }
#endif // P_IPT
#undef P_IPT
#undef PTTWAC_REMAP
  }
}

// Transformation 100, or ABb to BAb
// limitations: b cannot exceed # of allowed threads in the system
// Launch A*B work-groups of b work-items
/*__kernel void transpose_100(__global float *input, 
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
}*/
// Transformation 100 - Shared memory tiling
// Used by both transformation 0100 and 100 (see below)
// Assumes:
//  get_local_size(0) == wavefront size;
//  get_local_size(1) == number of warps
#define WARP_SIZE 32
//#define WARPS 6
#define P_IPT 0
void _transpose_100(__global float *input,
    int A, int B, int b, __global int *finished, volatile __local float *data,
    volatile __local float *backup, volatile __local int *done, const int warp_size) {
  int m = A*B-1;
  int tid = get_local_id(0);
  int group_id = get_group_id(0);
  int warp_id = get_local_id(1);
  int warps_group = get_local_size(1);
  //const int warp_size = WARP_SIZE; //get_local_size(0);

  // Recalculate IDs if virtual warp is used
  if (warp_size == 16){
    tid = get_local_id(0) & 15;
    int vwarps_in_warp = WARP_SIZE / warp_size;
    warps_group = warps_group * vwarps_in_warp;
    int vwarp_id = get_local_id(0) >> 4;
    warp_id = warp_id * vwarps_in_warp + vwarp_id;
  }
  else if (warp_size == 8){
    tid = get_local_id(0) & 7;
    int vwarps_in_warp = WARP_SIZE / warp_size;
    warps_group = warps_group * vwarps_in_warp;
    int vwarp_id = get_local_id(0) >> 3;
    warp_id = warp_id * vwarps_in_warp + vwarp_id;
  }
  else if (warp_size == 4){
    tid = get_local_id(0) & 3;
    int vwarps_in_warp = WARP_SIZE / warp_size;
    warps_group = warps_group * vwarps_in_warp;
    int vwarp_id = get_local_id(0) >> 2;
    warp_id = warp_id * vwarps_in_warp + vwarp_id;
  }
  else if (warp_size == 2){
    tid = get_local_id(0) & 1;
    int vwarps_in_warp = WARP_SIZE / warp_size;
    warps_group = warps_group * vwarps_in_warp;
    int vwarp_id = get_local_id(0) >> 1;
    warp_id = warp_id * vwarps_in_warp + vwarp_id;
  }

if(tid < b){
  for(int gid = group_id * warps_group + warp_id; gid < m;
    gid += get_num_groups(0) * warps_group) {
    int next_in_cycle = (gid * A)-m*(gid/B);
    if (next_in_cycle == gid)
      continue;

#if P_IPT
    for (;next_in_cycle > gid;
      next_in_cycle = (next_in_cycle*A)-m*(next_in_cycle/B))
      ;
    if (next_in_cycle !=gid)
      continue;
    for(int i = tid; i < b; i += warp_size){
      data[warp_id*b+i] = input[gid*b+i];
    }
    for (next_in_cycle = (gid * A)-m*(gid/B);
      next_in_cycle > gid;
      next_in_cycle = (next_in_cycle*A)-m*(next_in_cycle/B)) {
      for(int i = tid; i < b; i += warp_size){
        backup[warp_id*b+i] = input[next_in_cycle*b+i];
      }
      for(int i = tid; i < b; i += warp_size){
        input[next_in_cycle*b+i] = data[warp_id*b+i];
      }
      for(int i = tid; i < b; i += warp_size){
        data[warp_id*b+i] = backup[warp_id*b+i];
      }
    }
    for(int i = tid; i < b; i += warp_size){
      input[gid*b+i] = data[warp_id*b+i];
    }

#else

    for(int i = tid; i < b; i += warp_size){
      data[warp_id*b+i] = input[gid*b+i];
    }
    if (tid == 0){
      //make sure the read is not cached 
      done[warp_id] = atom_or(finished+gid, (int)0); 
    }

    for (;done[warp_id] == 0; 
        next_in_cycle = (next_in_cycle*A)-m*(next_in_cycle/B)) {
      for(int i = tid; i < b; i += warp_size){
        backup[warp_id*b+i] = input[next_in_cycle*b+i];
      }
      if (tid == 0) {
        done[warp_id] = atom_xchg(finished+next_in_cycle, (int)1);
      }
      if (!done[warp_id]) {
        for(int i = tid; i < b; i += warp_size){
          input[next_in_cycle*b+i] = data[warp_id*b+i];
        }
      }
      for(int i = tid; i < b; i += warp_size){
        data[warp_id*b+i] = backup[warp_id*b+i];
      }
    }
#endif
  }
}
}

void _transpose_100_b(__global float *input,
    int A, int B, int b, __global int *finished, volatile __local float *data,
    volatile __local float *backup, volatile __local int *done, const int warp_size) {
  int m = A*B-1;
  int tid = get_local_id(0);
  int group_id = get_group_id(0);

if(tid < b){
  for(int gid = group_id; gid < m; gid += get_num_groups(0)) {
    int next_in_cycle = (gid * A)-m*(gid/B);
    if (next_in_cycle == gid)
      continue;

    for(int i = tid; i < b; i += get_local_size(0)){
      data[i] = input[gid*b+i];
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    if (tid == 0){
      //make sure the read is not cached 
      done[0] = atom_or(finished+gid, (int)0); 
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    for (;done[0] == 0; 
        next_in_cycle = (next_in_cycle*A)-m*(next_in_cycle/B)) {
      for(int i = tid; i < b; i += get_local_size(0)){
        backup[i] = input[next_in_cycle*b+i];
      }
      barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
      if (tid == 0) {
        done[0] = atom_xchg(finished+next_in_cycle, (int)1);
      }
      barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
      if (!done[0]) {
        for(int i = tid; i < b; i += get_local_size(0)){
          input[next_in_cycle*b+i] = data[i];
        }
      }
      //barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
      for(int i = tid; i < b; i += get_local_size(0)){
        data[i] = backup[i];
      }
      //barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    }
  }
}
}

// Transformation 100 
__kernel void transpose_100(__global float *input,
    int A, int B, int b, __global int *finished, volatile __local float *data,
    volatile __local float *backup, int warp_size, volatile __local int *done) {
#if P_IPT
    _transpose_100(input, A, B, b, finished, data, backup, 0, WARP_SIZE);
#else
    //volatile __local int done[WARPS];
    _transpose_100(input, A, B, b, finished, data, backup, done, warp_size);
#endif

}

// Transformation 0100, or AaBb to ABab
__kernel void transpose_0100(__global float *input,
    int A, int B, int b, __global int *finished, volatile __local float *data,
    volatile __local float *backup, int warp_size, volatile __local int *done) {
  // for supporting transformation 0100
  finished += get_group_id(2) * A * B;
  input += get_group_id(2) * A * B * b;
#if P_IPT
  _transpose_100(input, A, B, b, finished, data, backup, 0, WARP_SIZE);
#else
  //volatile __local int done[WARPS];
  _transpose_100(input, A, B, b, finished, data, backup, done, warp_size);
#endif
}

#undef P_IPT

// Transformation 100 - Block-centric with shared memory tiling
__kernel void transpose_100_b(__global float *input,
    int A, int B, int b, __global int *finished, volatile __local float *data,
    volatile __local float *backup, int warp_size, volatile __local int *done) {
    _transpose_100_b(input, A, B, b, finished, data, backup, done, warp_size);

}

// Transformation 0100, or AaBb to ABab - Block-centric with shared memory tiling
__kernel void transpose_0100_b(__global float *input,
    int A, int B, int b, __global int *finished, volatile __local float *data,
    volatile __local float *backup, int warp_size, volatile __local int *done) {
  // for supporting transformation 0100
  finished += get_group_id(2) * A * B;
  input += get_group_id(2) * A * B * b;
  _transpose_100_b(input, A, B, b, finished, data, backup, done, warp_size);
}

