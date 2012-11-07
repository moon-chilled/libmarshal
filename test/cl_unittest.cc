#include "local_cl.hpp"
#include <gtest/gtest.h>
#include <cstdlib>
#include <string>
#include <vector>
#include <math.h>
#include "cl_marshal.h"
#include "plan.hpp"
namespace {
class libmarshal_cl_test : public ::testing::Test {
 public:
  cl_uint GetCtxRef(void) const {
    cl_uint rc;
    rc = context_->getInfo<CL_CONTEXT_REFERENCE_COUNT>();
    return rc;
  }
  cl_uint GetQRef(void) const {
    cl_uint rc = queue_->getInfo<CL_QUEUE_REFERENCE_COUNT>();
    return rc;
  }
 protected:
  virtual void SetUp(void);
  virtual void TearDown(void);
  libmarshal_cl_test() {}
  std::string device_name_;
  cl::Context *context_;
  cl::CommandQueue *queue_;
};

void libmarshal_cl_test::SetUp(void) {
  cl_int err;

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (platforms.size() == 0) {
    std::cerr << "Platform size 0\n";
    return;
  }
  int i = 0;
  for (; i < platforms.size(); i++) {
    std::vector<cl::Device> devices;
    platforms[i].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.size() == 0)
      continue;
    else
      break;
  }
  if (i == platforms.size()) {
    std::cerr << "None of the platforms have GPU\n";
    return;
  }
  cl_context_properties properties[] = 
  { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[i])(), 0};

  context_ = new cl::Context(CL_DEVICE_TYPE_GPU, properties, NULL, NULL, &err);
  queue_ = NULL;
  ASSERT_EQ(err, CL_SUCCESS);
  std::vector<cl::Device> devices = context_->getInfo<CL_CONTEXT_DEVICES>();
  // Get name of the devices
  devices[0].getInfo(CL_DEVICE_NAME, &device_name_);
  std::cerr << "Testing on device " << device_name_ << std::endl;

  // Create a command queue on the first GPU device
  queue_ = new cl::CommandQueue(*context_, devices[0]);
}

extern "C" void cl_marshal_finalize(void);
void libmarshal_cl_test::TearDown(void) {
  cl_marshal_finalize();
  delete queue_;
  delete context_;
}

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
//[w][h/t][t] to [h/t][w][t] 
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

}

TEST_F(libmarshal_cl_test, bug537) {
  int ws[6] = {40, 62, 197, 215, 59, 39};
  int hs[6] = {11948, 17281, 35588, 44609, 90449, 49152};
  for (int i = 0; i < 6; i++)
  for (int t = 16; t <= 64; t*=2) {
    int w = ws[i];
    int h = (hs[i]+t-1)/t*t;
    float *src = (float*)malloc(sizeof(float)*h*w);
    float *dst = (float*)malloc(sizeof(float)*h*w);
    float *dst_gpu = (float*)malloc(sizeof(float)*h*w);
    generate_vector(src, h*w);
    cpu_soa_asta(src, dst, h, w, t);
    cl_int err;
    cl::Buffer d_dst = cl::Buffer(*context_, CL_MEM_READ_WRITE,
        sizeof(float)*h*w, NULL, &err);
    ASSERT_EQ(err, CL_SUCCESS);
    cl_uint oldqref = GetQRef();
    ASSERT_EQ(queue_->enqueueWriteBuffer(
          d_dst, CL_TRUE, 0, sizeof(float)*h*w, src), CL_SUCCESS);
    cl_uint oldref = GetCtxRef();
    bool r = cl_soa_asta_pttwac((*queue_)(), d_dst(), h, w, t);
    EXPECT_EQ(oldref, GetCtxRef());
    EXPECT_EQ(oldqref, GetQRef());
    ASSERT_EQ(false, r);
    ASSERT_EQ(queue_->enqueueReadBuffer(d_dst, CL_TRUE, 0, sizeof(float)*h*w,
          dst_gpu), CL_SUCCESS);
    EXPECT_EQ(0, compare_output(dst_gpu, dst, h*w));
    Transposition tx(w,h/t);

    std::cerr << "Num cycles:"<<tx.GetNumCycles()<< "; percentage = " <<
      (float)tx.GetNumCycles()/(float)(h*w/t)*100 << "\n";
    free(src);
    free(dst);
    free(dst_gpu);
  }
}

TEST_F(libmarshal_cl_test, bug536) {
  int ws[6] = {40, 62, 197, 215, 59, 39};
  int hs[6] = {11948, 17281, 35588, 44609, 90449, 49152};
  for (int i = 0; i < 6; i++)
  for (int t = 16; t <= 64; t*=2) {
    int w = ws[i];
    int h = (hs[i]+t-1)/t*t;

    float *src = (float*)malloc(sizeof(float)*h*w);
    float *dst = (float*)malloc(sizeof(float)*h*w);
    float *dst_gpu = (float*)malloc(sizeof(float)*h*w);
    generate_vector(src, h*w);
    cpu_aos_asta(src, dst, h, w, t);
    cl_int err;
    cl::Buffer d_dst = cl::Buffer(*context_, CL_MEM_READ_WRITE,
        sizeof(float)*h*w, NULL, &err);
    ASSERT_EQ(err, CL_SUCCESS);
    ASSERT_EQ(queue_->enqueueWriteBuffer(
          d_dst, CL_TRUE, 0, sizeof(float)*h*w, src), CL_SUCCESS);
    cl_uint oldref = GetCtxRef();
    cl_uint oldqref = GetQRef();
    bool r = cl_aos_asta_pttwac((*queue_)(), d_dst(), h, w, t);
    EXPECT_EQ(oldref, GetCtxRef());
    EXPECT_EQ(oldqref, GetQRef());
    ASSERT_EQ(false, r);
    ASSERT_EQ(queue_->enqueueReadBuffer(d_dst, CL_TRUE, 0, sizeof(float)*h*w,
          dst_gpu), CL_SUCCESS);
    EXPECT_EQ(0, compare_output(dst_gpu, dst, h*w));
    free(src);
    free(dst);
    free(dst_gpu);
  }
}

TEST_F(libmarshal_cl_test, bug533) {
  int w = 20;
  for (int t=16; t<34; t++) {
    int h = (100*100*130+t-1)/t*t;
    float *src = (float*)malloc(sizeof(float)*h*w);
    float *dst = (float*)malloc(sizeof(float)*h*w);
    float *dst_gpu = (float*)malloc(sizeof(float)*h*w);
    generate_vector(src, h*w);
    cpu_aos_asta(src, dst, h, w, t);
    cl_int err;
    cl::Buffer d_dst = cl::Buffer(*context_, CL_MEM_READ_WRITE,
	sizeof(float)*h*w, NULL, &err);
    ASSERT_EQ(err, CL_SUCCESS);
    ASSERT_EQ(queue_->enqueueWriteBuffer(
	  d_dst, CL_TRUE, 0, sizeof(float)*h*w, src), CL_SUCCESS);
    cl_uint oldref = GetCtxRef();
    cl_uint oldqref = GetQRef();
    bool r = cl_aos_asta_bs((*queue_)(), d_dst(), h, w, t);
    EXPECT_EQ(oldref, GetCtxRef());
    EXPECT_EQ(oldqref, GetQRef());
    ASSERT_EQ(false, r);
    ASSERT_EQ(queue_->enqueueReadBuffer(d_dst, CL_TRUE, 0, sizeof(float)*h*w,
	  dst_gpu), CL_SUCCESS);
    EXPECT_EQ(0, compare_output(dst_gpu, dst, h*w));

    free(src);
    free(dst);
    free(dst_gpu);
  }
}

void tile(int x) {
  Factorize f(x);
  std::cout << "factors = ";
  for (std::map<int, int>::const_iterator it = f.get_factors().begin(), e = f.get_factors().end();
    it != e; it++) {
    std::cout << it->first << "^" << it->second<<"*";
  }
  std::cout << "\n";
  f.tiling_options();
  std::cout << "\n";
}

TEST_F(libmarshal_cl_test, tiles) {
  int ws[6] = {40, 62, 197, 215, 59, 39};
  int hs[6] = {11948, 17281, 35588, 44609, 90449, 49152};
  for (int n = 0; n < 6; n++) {
  int w = ws[n];
  int h = hs[n]; //(hs[0]+t-1)/t*t;

  float *src = (float*)malloc(sizeof(float)*h*w);
  float *dst = (float*)malloc(sizeof(float)*h*w);
  float *dst_gpu = (float*)malloc(sizeof(float)*h*w);
  generate_vector(src, h*w);

  Factorize hf(h), wf(w);
  hf.tiling_options();
  wf.tiling_options();
  std::vector<int> hoptions = hf.get_tile_sizes();
  std::vector<int> woptions = wf.get_tile_sizes();
  for (int i = 0 ; i < hoptions.size(); i++) {
    int A = h/hoptions[i], a = hoptions[i];
    for (int j = 0; j < woptions.size(); j++) {
      int B = w/woptions[j], b = woptions[j];
      std::cerr << " A = " << A << "; a= " << a << "; B = " << B << "; b= " << b <<"\n";
      cl_int err;
      cl::Buffer d_dst = cl::Buffer(*context_, CL_MEM_READ_WRITE,
          sizeof(float)*h*w, NULL, &err);
      ASSERT_EQ(err, CL_SUCCESS);
      ASSERT_EQ(queue_->enqueueWriteBuffer(
            d_dst, CL_TRUE, 0, sizeof(float)*h*w, src), CL_SUCCESS);
      bool r = false;
      r = cl_transpose((*queue_)(), d_dst(), A, a, B, b);
      // This may fail
      EXPECT_EQ(false, r);
      if (r != false)
        continue;
      // compute golden
      // [h/t][t][w] to [h/t][w][t]
      cpu_aos_asta(src, dst, h, w, a);
      // [h/t][w][t] to [h/t][t][w]
      cpu_soa_asta(dst, src, w*a, A, a);
      ASSERT_EQ(queue_->enqueueReadBuffer(d_dst, CL_TRUE, 0, sizeof(float)*h*w,
            dst_gpu), CL_SUCCESS);
      EXPECT_EQ(0, compare_output(dst_gpu, src, h*w));
    }
  }
  free(src);
  free(dst);
  free(dst_gpu);
  }
}

// testing 0100 transformation AaBb->ABab
TEST_F(libmarshal_cl_test, test_0100) {
  int bs[] = {32};
  int Bs[] = {57};
  int as[] = {62};
  int As[] = {128};
  int b = bs[0];
  int B = Bs[0];
  int a = as[0];
  int A = As[0];
  size_t size = A*a*B*b;

  float *src = (float*)malloc(sizeof(float)*size);
  float *dst = (float*)malloc(sizeof(float)*size);
  float *dst_gpu = (float*)malloc(sizeof(float)*size);
  generate_vector(src, size);

  cl_int err;
  cl::Buffer d_dst = cl::Buffer(*context_, CL_MEM_READ_WRITE,
      sizeof(float)*size, NULL, &err);
  ASSERT_EQ(err, CL_SUCCESS);
  ASSERT_EQ(queue_->enqueueWriteBuffer(
        d_dst, CL_TRUE, 0, sizeof(float)*size, src), CL_SUCCESS);
  bool r = false;
  r = cl_transpose_0100((*queue_)(), d_dst(), A, a, B, b);
  // This may fail
  EXPECT_EQ(false, r);
  // compute golden: A instances of aBb->Bab transformation
  for (int i = 0; i < A; i++) {
    cpu_soa_asta(src+i*(a*B*b),
      dst+i*(a*B*b), B*b /*h*/, a /*w*/, b /*t*/);
  }
  ASSERT_EQ(queue_->enqueueReadBuffer(d_dst, CL_TRUE, 0,
    sizeof(float)*size, dst_gpu), CL_SUCCESS);
  EXPECT_EQ(0, compare_output(dst_gpu, dst, size));
  free(src);
  free(dst);
  free(dst_gpu);
}

TEST(libmarshal_plan_test, cycle) {
  Transposition tx(2,3), tx2(3,5);
  EXPECT_EQ(1, tx.GetNumCycles());
  EXPECT_EQ(2, tx2.GetNumCycles());
}
