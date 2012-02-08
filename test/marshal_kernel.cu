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

__global__ static void PTTWAC_marshal(float *input, int tile_size, int width,
    clock_t *timer) {
  extern __shared__ unsigned finished[];
  int tidx = threadIdx.x;
  int m = tile_size*width - 1;
  input += blockIdx.x*tile_size*width;
  for (int id = tidx ; id < (tile_size * width + 31) / 32; id += blockDim.x) {
    finished[id] = 0;
  }
  __syncthreads();
  for (;tidx < tile_size*width; tidx += blockDim.x) {
    int next = (tidx * tile_size) % m;
    if (tidx != m && next != tidx) {
      float data1 = input[tidx];
      unsigned int mask = (1 << (tidx % 32));
      unsigned int flag_id = (((unsigned int) tidx) >> 5);
      int done = atomicOr(finished+flag_id, 0);
      done = (done & mask);
      for (; done == 0; next = (next * tile_size) % m) {
        float data2 = input[next];
        mask = (1 << (next % 32));
        flag_id = (((unsigned int)next) >> 5);
        done = atomicOr(finished+flag_id, mask);
        done = (done & mask);
        if (done == 0) {
          input[next] = data1;
        }
        data1 = data2;
      }
    }
  }
}
#endif //_LIBMARSHAL_KERNEL_CU_
