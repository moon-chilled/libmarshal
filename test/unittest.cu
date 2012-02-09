#include <gtest/gtest.h>
#include <cstdlib>
#include "marshal.h"
//namespace {
class libmarshal_test : public ::testing::Test {
 protected:
  virtual void SetUp(void) {}
  virtual void TearDown(void) {}
  libmarshal_test() {}
//};


int compare_output(float *output, float *ref, int dim) {
  int pass = 1;
  int i;
  for (i = 0; i < dim; i++) {
    float diff = fabs(ref[i] - output[i]);
    if ((diff - 0.0f) > 0.00001f && diff > 0.01*fabs(ref[i])) {
      printf("line: %d ref: %f actual: %f diff: %f\n",
          i, ref[i], output[i], diff);
      pass = 0;
      break;
    }
  }
#if 0
  printf("\n");
  if (pass)
    printf("comparison passed\n");
  else
    printf("comparison failed\n");
  printf("\n");
#endif
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

void cpu_soa_asta(float *src, float *dst, int height, int width,
    int tile_size) {
  // We only support height == multiple of tile size
  assert((height/tile_size)*tile_size == height);
  for (int k = 0; k < width; k++) {
    for (int i = 0; i<height/tile_size; i++) { //For all tiles
      for(int j = 0; j < tile_size; j++) {
        //from src[k][i][j] to dst[i][k][j]
        dst[i*width*tile_size + k*tile_size + j] =
          src[k*height+i*tile_size + j];
      }
    }
  }
}

};

TEST_F(libmarshal_test, bug528) {
  int h = 16;
  int t = 4;
  int w = 3;
  float *src = (float*)malloc(sizeof(float)*h*w);
  float *dst = (float*)malloc(sizeof(float)*h*w);
  float *dst_gpu = (float*)malloc(sizeof(float)*h*w);
  generate_vector(src, h*w);
  cpu_soa_asta(src, dst, h, w, t);
  float *d_dst;
  cudaMalloc(&d_dst, sizeof(float)*h*w);
  cudaMemcpy(d_dst, src, sizeof(float)*h*w, cudaMemcpyHostToDevice);
  bool r = gpu_soa_asta_pttwac(d_dst, h, w, t, NULL);
  ASSERT_EQ(false, r);

  cudaMemcpy(dst_gpu, d_dst, sizeof(float)*h*w, cudaMemcpyDeviceToHost);
  EXPECT_EQ(0, compare_output(dst_gpu, dst, h*w));
  free(src);
  free(dst);
  cudaFree(d_dst);
}

TEST_F(libmarshal_test, bug525) {
  int h = 16*1024;
  int t = 16;
  for (int w = 1; w < 100; w++) {
    float *src = (float*)malloc(sizeof(float)*h*w);
    float *dst = (float*)malloc(sizeof(float)*h*w);
    float *dst_gpu = (float*)malloc(sizeof(float)*h*w);
    generate_vector(src, h*w);
    cpu_aos_asta(src, dst, h, w, t);

    float *d_dst;
    cudaMalloc(&d_dst, sizeof(float)*h*w);
    cudaMemcpy(d_dst, src, sizeof(float)*h*w, cudaMemcpyHostToDevice);
    bool r = gpu_aos_asta_pttwac(d_dst, h, w, t, NULL);
    ASSERT_EQ(false, r);

    cudaMemcpy(dst_gpu, d_dst, sizeof(float)*h*w, cudaMemcpyDeviceToHost);

    EXPECT_EQ(0, compare_output(dst_gpu, dst, h*w));
    free(src);
    free(dst);
    cudaFree(d_dst);
  }
}

TEST_F(libmarshal_test, bug524) {
  int h = 16*1024;
  int w = 66;
  int t = 16;
  float *src = (float*)malloc(sizeof(float)*h*w);
  float *dst = (float*)malloc(sizeof(float)*h*w);
  float *dst_gpu = (float*)malloc(sizeof(float)*h*w);
  generate_vector(src, h*w);
  cpu_aos_asta(src, dst, h, w, t);

  float *d_dst;
  cudaMalloc(&d_dst, sizeof(float)*h*w);
  cudaMemcpy(d_dst, src, sizeof(float)*h*w, cudaMemcpyHostToDevice);
  bool r = gpu_aos_asta_pttwac(d_dst, h, w, t, NULL);
  ASSERT_EQ(false, r);
  
  cudaMemcpy(dst_gpu, d_dst, sizeof(float)*h*w, cudaMemcpyDeviceToHost);

  EXPECT_EQ(0, compare_output(dst_gpu, dst, h*w));
  free(src);
  free(dst);
  cudaFree(d_dst);
}

TEST_F(libmarshal_test, bug523) {
  int h = 16*1024;
  int w = 6;
  int t = 16;
  float *src = (float*)malloc(sizeof(float)*h*w);
  float *dst = (float*)malloc(sizeof(float)*h*w);
  float *dst_gpu = (float*)malloc(sizeof(float)*h*w);
  generate_vector(src, h*w);
  cpu_aos_asta(src, dst, h, w, t);

  float *d_dst;
  cudaMalloc(&d_dst, sizeof(float)*h*w);
  cudaMemcpy(d_dst, src, sizeof(float)*h*w, cudaMemcpyHostToDevice);
  bool r = gpu_aos_asta_bs(d_dst, h, w, t, NULL);
  ASSERT_EQ(false, r);
  
  cudaMemcpy(dst_gpu, d_dst, sizeof(float)*h*w, cudaMemcpyDeviceToHost);

  EXPECT_EQ(0, compare_output(dst_gpu, dst, h*w));
  free(src);
  free(dst);
  cudaFree(d_dst);
}
