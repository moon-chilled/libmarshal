#include <stdlib.h>
#include "marshal_kernel.cu"

extern "C" void gpu_aos_asta(float *src, int height, int width, int tile_size, clock_t *timer) {
  //BS_marshal<<<10, 10>>>(input, timer);
}
