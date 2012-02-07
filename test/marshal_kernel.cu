#ifndef _LIBMARSHAL_KERNEL_CU_
#define _LIBMARSHAL_KERNEL_CU_

__global__ static void BS_marshal(float *input, clock_t *timer) {
  clock_t time1 = clock();
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  __syncthreads();
  if (tid == 0)
    timer[bid] = clock() - time1;
}

#endif //_LIBMARSHAL_KERNEL_CU_
