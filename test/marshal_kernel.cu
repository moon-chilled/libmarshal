#ifndef _LIBMARSHAL_KERNEL_CU_
#define _LIBMARSHAL_KERNEL_CU_

__global__ static void BS_marshal(float *input,
    int tile_size, int width, clock_t *timer) {
//  clock_t time1 = clock();
  int tidx = threadIdx.x;
  int tidy = threadIdx.y;
  int bid = blockIdx.x;
  input += tile_size*width*bid;
  float tmp = input[tidy*width+tidx];
  __syncthreads();
  __threadfence();
  input[tidx*tile_size+tidy] = tmp;
#if 0
  if (tid == 0)
    timer[bid] = clock() - time1;
#endif
}

#endif //_LIBMARSHAL_KERNEL_CU_
