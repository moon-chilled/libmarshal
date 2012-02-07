#include <cstdlib>
#include <cassert>
#include "marshal.h"
#include "marshal_kernel.cu"
extern "C" void gpu_aos_asta(float *src, int height, int width,
    int tile_size, clock_t *timer) {
  assert ((height/tile_size)*tile_size == height);
  dim3 threads (width, tile_size, 1);
  BS_marshal<<<height/tile_size, threads>>>(src, tile_size, width, timer);
}
