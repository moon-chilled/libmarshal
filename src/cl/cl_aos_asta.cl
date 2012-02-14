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
__kernel void PTTWAC_marshal_soa(__global float *input, int tile_size,
    int width, __global int *finished) {
  int m = get_num_groups(0)-1;
  int tid = get_local_id(0);
  float data;
  int gid = get_group_id(0);
  if (gid == m)
    return;

  int next_in_cycle = (gid * width)%m;
  if (next_in_cycle == gid)
    return;

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

#if 0
__kernel void TransposeEllF_cycle (
  __global float *data,
  __global int *col_idx,
  int dimx,
  __local volatile int *finished 
) {
  int m = TILE*dimx - 1;
  int base_idx = get_group_id(0)*(TILE*dimx);
  int tid = get_local_id(0);
  finished[tid] = 0;
  barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
  if (tid == m)
    return;
  int next = (tid*TILE)%m;
  if (next == tid)
    return;

  float data1 = data[base_idx+tid];
  int col_idx1 = col_idx[base_idx+tid];
  int done = atom_or(&finished[tid],0);

  for (;done == 0; next = (next*TILE)%m) {
    float data2 = data[base_idx+next];
    int col_idx2 = col_idx[base_idx+next];
    done = atom_xchg(&finished[next], (int)1); 
    if (done==0) {
      data[base_idx+next] = data1;
      col_idx[base_idx+next] = col_idx1;
    }
    data1 = data2;
    col_idx1 = col_idx2;
  }
}

__kernel void TransposeEllF_cycle_coarsen (
  __global float *data,
  __global int *col_idx,
  int dimx,
  __local volatile unsigned int *finished 
) {
  int m = TILE*dimx - 1;
  int base_idx = get_group_id(0)*(TILE*dimx);
  int tid;
  //int i;
  for (tid = get_local_id(0); tid < TILE*dimx/32; tid += get_local_size(0)) {
    finished[tid] = 0;
  }
  //tid = get_local_id(0);
  barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
  //for (i = 0; i < dimx; i++) {
  for (tid = get_local_id(0); tid < TILE*dimx; tid += get_local_size(0)) {
    int next = (tid*TILE)%m;
    if (tid != m && next != tid) {
      float data1 = data[base_idx+tid];
      int col_idx1 = col_idx[base_idx+tid];
      //int mask = 0x1 << ((tid & 0x3e0)>>5);
      //int flag_id = ((tid & 0xfffffc00) >> 5) + tid%32;
      //int done = atom_or(&finished[flag_id],0);
      //done = done & mask;
      unsigned int mask = (0x1 << (tid % 32));
      unsigned int flag_id = (((unsigned int)tid) >> 5); 
      int done = atom_or(&finished[flag_id],0);
      done = (done & mask);
      //unsigned int done = atom_or(&finished[tid],0);

      for (;done == 0; next = (next*TILE)%m) {
        float data2 = data[base_idx+next];
        int col_idx2 = col_idx[base_idx+next];
        //mask = 0x1 << ((next & 0x3e0)>>5);
        //flag_id = ((next & 0xfffffc00) >> 5) + next%32;
        //done = atom_xchg(&finished[flag_id], mask); 
        //done = done & mask;
        mask = (0x1 << (next % 32));
        flag_id = (((unsigned int)next) >> 5);
        done = atom_or(&finished[flag_id], mask); 
        done = (done & mask);
        //done = atom_xchg(&finished[next], (int)1); 
        if (done==0) {
          data[base_idx+next] = data1;
          col_idx[base_idx+next] = col_idx1;
        }
        data1 = data2;
        col_idx1 = col_idx2;
      }
    }
    //tid += TILE;
  }
}


__kernel void TransposeEllR (
  __global float *data,
  __global int *col_idx,
  int dimx,
  int dimxy
) {
  float data_buf0;
  float data_buf1;
  float data_buf2;
  float data_buf3;
  int idx_buf0;
  int idx_buf1;
  int idx_buf2;
  int idx_buf3;
  int tid = get_local_id(0);
  int gid = get_group_id(0)*(get_local_size(0)<<LOG_COARSEN) + tid;
  int row = gid/dimx;
  int col = gid%dimx;
  if (gid < dimxy) {
    data_buf0 = data[__FF2D(dimx,row,col)];
    idx_buf0 = col_idx[__FF2D(dimx,row,col)];
  }
  //coarsen by 4
  gid += get_local_size(0);
  row += get_local_size(0)/dimx;
  if (gid < dimxy) {
    data_buf1 = data[__FF2D(dimx,row,col)];
    idx_buf1 = col_idx[__FF2D(dimx,row,col)];
  }
  gid += get_local_size(0);
  row += get_local_size(0)/dimx;
  if (gid < dimxy) {
    data_buf2 = data[__FF2D(dimx,row,col)];
    idx_buf2 = col_idx[__FF2D(dimx,row,col)];
  }
  gid += get_local_size(0);
  row += get_local_size(0)/dimx;
  if (gid < dimxy) {
    data_buf3 = data[__FF2D(dimx,row,col)];
    idx_buf3 = col_idx[__FF2D(dimx,row,col)];
  }

  barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

  gid = get_group_id(0)*(get_local_size(0)<<LOG_COARSEN) + tid;
  if (gid < dimxy) {
    data[gid] = data_buf0;
    col_idx[gid] = idx_buf0;
  }
  //coarsen by 4
  gid += get_local_size(0);
  if (gid < dimxy) {
    data[gid] = data_buf1;
    col_idx[gid] = idx_buf1;
  }
  gid += get_local_size(0);
  if (gid < dimxy) {
    data[gid] = data_buf2;
    col_idx[gid] = idx_buf2;
  }
  gid += get_local_size(0);
  if (gid < dimxy) {
    data[gid] = data_buf3;
    col_idx[gid] = idx_buf3;
  }
}
  

__kernel void spmv_ell (
  __global float *data,
  __global int *col_idx,
  __global float *vector_i,
  __global float *vector_o,
  int dimx_ell
) {
  int row = get_global_id(0);
  float r = 0;
  for (int col = 0; col < dimx_ell; col++) {
    int idx = col_idx[__FF2D(dimx_ell,row,col)];
    if (idx >= 0)
      r += data[__FF2D(dimx_ell,row,col)]*vector_i[idx];
  }
  vector_o[row] = r;
}
#endif
