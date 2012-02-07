#include <gtest/gtest.h>
#include <cstdlib>

namespace {
class libmarshal_test : public ::testing::Test {
 protected:
  virtual void SetUp(void) {}
  virtual void TearDown(void) {}
  libmarshal_test() {}
};


int compare_output(float *output, float *ref, int dim) {
  int pass = 1;
  int i;
  for (i = 0; i < dim; i++) {
    float diff = fabs(ref[i] - output[i]);
    if ((diff - 0.0f) > 0.00001f && diff > 0.01*fabs(ref[i])) {
      printf("line : %d ref: %f actual: %f diff: %f\n", i, ref[i], output[i], diff);
      pass = 0;
      break;
    }
  }
  printf("\n");
  if (pass)
    printf("comparison passed\n");
  else
    printf("comparison failed\n");
  printf("\n");
  return pass != 1;
}

// Generate a matrix of random numbers
int generate_vector(float *x_vector, int dim) 
{       
  srand(5432);
  for(int i=0;i<dim;i++) {
    x_vector[i] = ((float) (rand() % 100) / 100);
  }
  return 0;
}

void cpu_aos_asta(float *src, float *dst, int height, int width,
    int tile_size) {
  // We only support height == multiple of tile size
  assert((height/tile_size)*tile_size == height);
  for (int i = 0; i<height/tile_size; i++) { //For all tiles
    float *src_start = src+i*tile_size*width;
    float *dst_start = dst+i*tile_size*width;
    for(int j = 0; j < tile_size; j++) {
      for (int k = 0; k < width; k++) {
        dst_start[j+k*tile_size]=src_start[j*width+k];
      }
    }
  }
}
};

extern "C" void gpu_aos_asta(float *src, int height, int width, int tile_size, clock_t *timer);

TEST_F(libmarshal_test, DISABLED_T1) {
  float *src = (float*)malloc(sizeof(float)*4*4*3);
  float *dst = (float*)malloc(sizeof(float)*4*4*3);
  float *dst_gpu = (float*)malloc(sizeof(float)*4*4*3);
  generate_vector(src, 4*4*3);
  cpu_aos_asta(src, dst, 4*4, 3, 4);

  float *d_dst;
  cudaMalloc(&d_dst, sizeof(float)*4*4*3);
  cudaMemcpy(d_dst, src, sizeof(float)*4*4*3, cudaMemcpyHostToDevice);
  gpu_aos_asta(dst_gpu, 4*4, 3, 4, NULL);
  cudaMemcpy(dst_gpu, d_dst, sizeof(float)*4*4*3, cudaMemcpyDeviceToHost);

  EXPECT_EQ(0, compare_output(dst, dst_gpu, 4*4*4));
  free(src);
  free(dst);

}
