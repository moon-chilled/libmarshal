#include "local_cl.hpp"
#include <gtest/gtest.h>
#include <cstdlib>
#include <string>
#include <vector>
#include <math.h>
#include <limits.h>
#include "cl_marshal.h"
#include "plan.hpp"
#include "/usr/include/gsl/gsl_sort.h"

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
  //int i = 1;
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
  queue_ = new cl::CommandQueue(*context_, devices[0], CL_QUEUE_PROFILING_ENABLE);
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
  //for (int t = 1; t <= 2048; t*=2) {	// 4096
  for (int t = 1; t <= 1024; t*=2) {
    int w = ws[i];
    int h = (hs[i]+t-1)/t*t;

    std::cerr << "w = "<<w<< "; h/t = " <<h/t<< "; t = " <<t<< ", ";

    float *src = (float*)malloc(sizeof(float)*h*w);
    float *dst = (float*)malloc(sizeof(float)*h*w);
    float *dst_gpu = (float*)malloc(sizeof(float)*h*w);
    generate_vector(src, h*w);
    cl_int err;
    cl::Buffer d_dst = cl::Buffer(*context_, CL_MEM_READ_WRITE,
        sizeof(float)*h*w, NULL, &err);
    ASSERT_EQ(err, CL_SUCCESS);
    cl_uint oldqref = GetQRef();
    ASSERT_EQ(queue_->enqueueWriteBuffer(
          d_dst, CL_TRUE, 0, sizeof(float)*h*w, src), CL_SUCCESS);
    cl_uint oldref = GetCtxRef();
    // Change N to something > 1 to compute average performance (and use some WARM_UP runs).
    const int N = 4;
    const int WARM_UP = 2;
    cl_ulong et = 0;
    for (int n = 0; n < N+WARM_UP; n++) {
      if (n == WARM_UP)
        et = 0;
      bool r = cl_transpose_100((*queue_)(), d_dst(), w, h/t, t, &et);
      EXPECT_EQ(oldref, GetCtxRef());
      EXPECT_EQ(oldqref, GetQRef());
      ASSERT_EQ(false, r);
      ASSERT_EQ(queue_->enqueueReadBuffer(d_dst, CL_TRUE, 0,
            sizeof(float)*h*w, dst_gpu), CL_SUCCESS);
      if ((n % 2) == 0) {
        cpu_soa_asta(src, dst, h, w, t);
        EXPECT_EQ(0, compare_output(dst_gpu, dst, h*w));
      } else {
        cpu_soa_asta(dst, src, h, w, t);
        EXPECT_EQ(0, compare_output(dst_gpu, src, h*w));
      }
    }
    Transposition tx(w,h/t);
    std::cerr << "Throughput = " << float(2*h*w*sizeof(float)*N)/et;
    std::cerr << " GB/s\t";
    std::cerr << "Num cycles:"<<tx.GetNumCycles()<< "; percentage = " <<
      (float)tx.GetNumCycles()/(float)(h*w/t)*100 << "\n";
    free(src);
    free(dst);
    free(dst_gpu);
  }
}

#define CHECK_RESULTS 1 
#define NVIDIA 1
#define SP 1 // SP = 1 -> Single Precision; SP = 0 -> Double Precision 

#if NVIDIA
// Shared memory in Fermi and Kepler is 48 KB, i.e., 12288 SP or 6144 DP.
#if SP
#define MAX_MEM 12288 // Use 4096 for other devices.
#else
#define MAX_MEM 6144 // Use 2048 for other devices.
#endif
#else
// Local memory in AMD devices is 32 KB, i.e., 8192 SP or 4096 DP.
#if SP
#define MAX_MEM 8192 // Use 4096 for other devices.
#else
#define MAX_MEM 4096 // Use 2048 for other devices.
#endif
#endif

TEST_F(libmarshal_cl_test, bug536) {
  int ws[6] = {40, 62, 197, 215, 59, 39};
  int hs[6] = {11948, 17281, 35588, 44609, 90449, 49152};
  for (int i = 0; i < 6; i++)
  for (int t = 16; t <= 64; t*=2) {
    int w = ws[i];
    int h = (hs[i]+t-1)/t*t;

#define P 1 // Padding size
    bool r;
    for(int S_f = 1; S_f <= 32; S_f *=2){ // (Use S_f <= 1 for testing IPT) - S_f = Spreading factor

      int sh_sz2 = S_f * ((t*w+31)/32);
      sh_sz2 += (sh_sz2 >> 5) * P;

      std::cerr << "w = " << w << ", t = " << t << ", w*t = " << w*t << ", S_f = " << S_f;
      std::cerr << ", P = " << P << ", " << (int)(5-log2(S_f)) << ", " << sh_sz2 << ", ";

      if(sh_sz2 > MAX_MEM) std::cerr << "\n";
      else{
        float *src = (float*)malloc(sizeof(float)*h*w);
        float *dst = (float*)malloc(sizeof(float)*h*w);
        float *dst_gpu = (float*)malloc(sizeof(float)*h*w);
        generate_vector(src, h*w);
        cl_int err;
        cl::Buffer d_dst = cl::Buffer(*context_, CL_MEM_READ_WRITE,
            sizeof(float)*h*w, NULL, &err);
        ASSERT_EQ(err, CL_SUCCESS);
        ASSERT_EQ(queue_->enqueueWriteBuffer(
            d_dst, CL_TRUE, 0, sizeof(float)*h*w, src), CL_SUCCESS);
        cl_uint oldref = GetCtxRef();
        cl_uint oldqref = GetQRef();
        cl_ulong et = 0;
        // Change N to something > 1 to compute average performance (and use some WARM_UP runs).
        const int N = 1;
        const int WARM_UP = 0;
        for (int n = 0; n < N+WARM_UP; n++) {
          if (n == WARM_UP)
            et = 0;
          r = cl_transpose_010_pttwac((*queue_)(), d_dst(), h/t, t, w, &et, S_f, P);
          EXPECT_EQ(oldref, GetCtxRef());
          EXPECT_EQ(oldqref, GetQRef());
          ASSERT_EQ(false, r);
          ASSERT_EQ(queue_->enqueueReadBuffer(d_dst, CL_TRUE, 0,
              sizeof(float)*h*w, dst_gpu), CL_SUCCESS);
          if ((n%2) == 0) {
            cpu_aos_asta(src, dst, h, w, t);
            EXPECT_EQ(0, compare_output(dst_gpu, dst, h*w));
          } else {
            cpu_aos_asta(dst, src, h, w, t);
            EXPECT_EQ(0, compare_output(dst_gpu, src, h*w));
          }
        }
        std::cerr << "Throughput = " << float(h*w*2*sizeof(float)*N) / et;
        std::cerr << " GB/s\t";

        Transposition tx(t,w);
        std::cerr << "Num cycles:"<<tx.GetNumCycles()<< "; percentage = " <<
          (float)tx.GetNumCycles()/(float)(w*t)*100 << "\n";

        free(src);
        free(dst);
        free(dst_gpu);
      }
    }
  }
}

TEST_F(libmarshal_cl_test, bug533) {
  for (int w = 3; w <= 768; w*=4)
  for (int t=1; t<=8; t+=1) {
    int h = (500/w+1)*(100*130+t-1)/t*t;
    std::cerr << "A = " << h/t << ", a = " << t << ", B = " << w << ", w*t = " << w*t << "\t";
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

#include "heuristic.cc"
#include "heuristic33.cc"

TEST_F(libmarshal_cl_test, full) {
#define RANDOM 1
#if RANDOM
  // Matrix sizes
  // For general matrices
#if SP
  // Single precision
  const int h_max = 20000; const int h_min = 1000;
  const int w_max = 20000; const int w_min = 1000;
#else
  // Double precision
  const int h_max = 10000; const int h_min = 1000;
  const int w_max = 10000; const int w_min = 1000;
  // For skinny matrices (AoS-SoA)
  //const int h_max = 10e7; const int h_min = 10000;
  //const int w_max = 32; const int w_min = 2;
#endif

  for (int n = 12; n < 16; n++){
  // Generate random dimensions
  srand(n+1);
  int h = rand() % (h_max-h_min) + h_min;
  int w = rand() % (w_max-w_min) + w_min;
#else 
  int ws[] = {1800, 2500, 3200, 3900, 5100, 7200}; // Matrix sizes in PPoPP2014 paper
  int hs[] = {7200, 5100, 4000, 3300, 2500, 1800};
  for (int n = 0; n < 6; n++) {
  int w = ws[n];
  int h = hs[n];
#endif

  std::cerr << "" << h << "," << w << "\t";

  std::vector<int> hoptions;
  std::vector<int> woptions;
  int pad_h, pad_w = 0;
  bool done_h, done_w = false;
  //printf("%d %d %s %s\n", pad_h, pad_w, done_h ? "true" : "false", done_w ? "true" : "false");
  do{
    // Factorize dimensions
    Factorize hf(h), wf(w);
    hf.tiling_options();
    wf.tiling_options();
    hoptions = hf.get_tile_sizes();
    woptions = wf.get_tile_sizes();
    std::cerr << "" << hoptions.size() << "," << woptions.size() << "\t";

    // Pad h
    if (hoptions.size() < 1){
      pad_h++;
      h++;
    }
    else done_h = true;

    // Pad w
    if (woptions.size() < 5){
      pad_w++;
      w++;
    }
    else done_w = true;

  }while(!done_h || !done_w);

  std::cerr << "percent_pad " << ((float)((w-(w-pad_w))*100)/(float)(w-pad_w)) << "%\t";
  std::cerr << "percent_unpad " << ((float)((h-(h-pad_h))*100)/(float)(h-pad_h)) << "%\t";

  float *src = (float*)malloc(sizeof(float)*h*w);
  float *dst = (float*)malloc(sizeof(float)*h*w);
  float *dst_gpu = (float*)malloc(sizeof(float)*h*w);
  generate_vector(src, (h - pad_h) * (w - pad_w));

#if 1
  // Sort factors
  //for(int x=0; x<hoptions.size(); x++) printf("%d ", hoptions[x]);
  //printf("\n");
  size_t hf_sorted[hoptions.size()];
  size_t wf_sorted[woptions.size()];
  gsl_sort_int_index((size_t *)hf_sorted, &hoptions[0], 1, hoptions.size());
  //for(int x=0; x<hoptions.size(); x++) printf("%d ", hoptions[hf_sorted[x]]);
  //printf("\n");
  gsl_sort_int_index((size_t *)wf_sorted, &woptions[0], 1, woptions.size());
  for(int x=0; x<woptions.size(); x++) printf("%d ", woptions[wf_sorted[x]]);
  printf("\n");
#endif

#define BRUTE 0
#if BRUTE
  // Brute-force search
  cl_ulong max_et = ULONG_MAX;
  for (int i = 0 ; i < hoptions.size(); i++) {
    int A = h/hoptions[hf_sorted[i]], a = hoptions[hf_sorted[i]];
    for (int j = 0; j < woptions.size(); j++) {
      int B = w/woptions[wf_sorted[j]], b = woptions[wf_sorted[j]];
#else
  // Heuristic for determining tile dimensions
  int A, a, B, b;
  Heuristic(&A, &a, &B, &b, hf_sorted, wf_sorted, hoptions, woptions, h, w);
  std::cerr << "" << A << "," << a << ",";
  std::cerr << "" << B << "," << b <<",";
#endif

  cl_int err;
  cl::Buffer d_dst = cl::Buffer(*context_, CL_MEM_READ_WRITE,
      sizeof(float)*h*w, NULL, &err);
  ASSERT_EQ(err, CL_SUCCESS);
  err = queue_->enqueueWriteBuffer(
        d_dst, CL_TRUE, 0, sizeof(float)*h*w, src);
  EXPECT_EQ(err, CL_SUCCESS);
  if (err != CL_SUCCESS)
    continue;

  bool r = false;
  cl_ulong et = 0;
  cl_ulong et2 = 0;
  cl_ulong et3 = 0;
  // Change N to something > 1 to compute average performance (and use some WARM_UP runs).
  const int N = 4; 
  const int WARM_UP = 2;

//if(a <= 1536 && b <= 1536){
    if((a >= 6 && a*B*b <= MAX_MEM) || b < 3 && ((a >= b && ((a*B*b+31)/32) + ((((a*B*b+31)/32)>>5)*1) <= MAX_MEM) || (A > b && ((A*B*b+31)/32) + ((((A*B*b+31)/32)>>5)*1) <= MAX_MEM))){
      /*if (b < 3 && ((a*B*b+31)/32) + ((((a*B*b+31)/32)>>5)*1) > MAX_MEM){
        int temp = A;
        A = a;
        a = temp;
      }*/
      Heuristic33(&A, &a, &B, &b, hf_sorted, wf_sorted, hoptions, woptions, h, w);

      std::cerr << "" << A << "," << a << ",";
      std::cerr << "" << B*b << ",";

      // Calculate spreading factor
      int S_f = MAX_MEM / (((a*B*b+31)/32) + ((((a*B*b+31)/32)>>5)*P));
      std::cerr << "S_f = " << S_f << ",";
      if (S_f < 2) S_f = 1;
      else if (S_f >=2 && S_f < 4) S_f = 2;
      else if (S_f >=4 && S_f < 8) S_f = 4;
      else if (S_f >=8 && S_f < 16) S_f = 8;
      else if (S_f >=16 && S_f < 32) S_f = 8; //16;
      else S_f = 32; // BS will be used
      std::cerr << "" << S_f << ",\t";

      // 2-stage approach
      for (int n = 0; n < N+WARM_UP; n++) {
        if (n == WARM_UP)
          et = 0;
        r = cl_transpose((*queue_)(), d_dst(), A, a, B, b, S_f, 2, &et); 
        // This may fail
        EXPECT_EQ(false, r);
        if (r != false)
          continue;
#if CHECK_RESULTS
        ASSERT_EQ(queue_->enqueueReadBuffer(d_dst, CL_TRUE, 0, sizeof(float)*h*(w-pad_w),
              dst_gpu), CL_SUCCESS);
        // compute golden
        // [h/t][t][w] to [h/t][w][t]
        cpu_aos_asta(src, dst, h, w-pad_w, a);
        // [h/t][w][t] to [h/t][t][w]
        cpu_soa_asta(dst, src, (w-pad_w)*a, A, a);
        EXPECT_EQ(0, compare_output(dst_gpu, src, h*(w-pad_w)));
#endif
      }
    }
    else if((b >= 6 && b*A*a <= MAX_MEM) || a < 3 && (b >= a && (((b*A*a+31)/32) + ((((b*A*a+31)/32)>>5)*1) <= MAX_MEM) || (B > a && ((B*A*a+31)/32) + ((((B*A*a+31)/32)>>5)*1) <= MAX_MEM))){
      /*if (a < 3 && ((b*A*a+31)/32) + ((((b*A*a+31)/32)>>5)*1) > MAX_MEM){
        int temp = B;
        B = b;
        b = temp;
      }*/
      Heuristic33(&B, &b, &A, &a, wf_sorted, hf_sorted, woptions, hoptions, w, h);

      std::cerr << "" << A*a << ",";
      std::cerr << "" << B << "," << b << ",";

      // Calculate spreading factor
      int S_f = MAX_MEM / (((b*A*a+31)/32) + ((((b*A*a+31)/32)>>5)*P));
      std::cerr << "S_f = " << S_f << ",";
      if (S_f < 2) S_f = 1;
      else if (S_f >=2 && S_f < 4) S_f = 2;
      else if (S_f >=4 && S_f < 8) S_f = 4;
      else if (S_f >=8 && S_f < 16) S_f = 8;
      else if (S_f >=16 && S_f < 32) S_f = 8; //16;
      else S_f = 32; // BS will be used
      std::cerr << "" << S_f << ",\t";

      // 2-stage approach
      for (int n = 0; n < N+WARM_UP; n++) {
        if (n == WARM_UP)
          et = 0;
        r = cl_transpose((*queue_)(), d_dst(), A, a, B, b, S_f, 22, &et);
        // This may fail
        EXPECT_EQ(false, r);
        if (r != false)
          continue;
#if CHECK_RESULTS
        ASSERT_EQ(queue_->enqueueReadBuffer(d_dst, CL_TRUE, 0, sizeof(float)*h*(w-pad_w),
              dst_gpu), CL_SUCCESS);
        // compute golden
        // [h/t][t][w] to [h/t][w][t]
        cpu_aos_asta(src, dst, h, w-pad_w, a);
        // [h/t][w][t] to [h/t][t][w]
        cpu_soa_asta(dst, src, (w-pad_w)*a, A, a);
        EXPECT_EQ(0, compare_output(dst_gpu, src, h*(w-pad_w)));
#endif
      }
    }
    else{
      std::cerr << "" << A << "," << a << ",";
      std::cerr << "" << B << "," << b <<",";

      // Calculate spreading factor
      int S_f = MAX_MEM / (((a*b+31)/32) + ((((a*b+31)/32)>>5)*P));
      std::cerr << "S_f = " << S_f << ",";
      if (S_f < 2) S_f = 1;
      else if (S_f >=2 && S_f < 4) S_f = 2;
      else if (S_f >=4 && S_f < 8) S_f = 4;
      else if (S_f >=8 && S_f < 16) S_f = 8;
      else if (S_f >=16 && S_f < 32) S_f = 8; //16;
      else S_f = 32; // BS will be used
      std::cerr << "" << S_f << ",\t";

      // 3-stage approach
      for (int n = 0; n < N+WARM_UP; n++) {
        if (n == WARM_UP){
          et = 0; et2 = 0; et3 = 0;
        }

        // Padding
        if (pad_w > 0){
          r = cl_padding((*queue_)(), d_dst(), w - pad_w, h - pad_h, w, &et2);
          // This may fail
          EXPECT_EQ(false, r);
          if (r != false)
            continue;
          //std::cerr << "Padding Throughput = " << float(2*h*w*sizeof(float))/et2;
          //std::cerr << " GB/s\t";
        }

        // Transpose
        if (a <= b)
          r = cl_transpose((*queue_)(), d_dst(), A, a, B, b, S_f, 3, &et);
        else
          r = cl_transpose((*queue_)(), d_dst(), A, a, B, b, S_f, 32, &et);
        // This may fail
        EXPECT_EQ(false, r);
        if (r != false)
          continue;

        // Unpadding
        if (pad_h > 0){
          r = cl_unpadding((*queue_)(), d_dst(), h - pad_h, w - pad_w, h, &et3);
          // This may fail
          EXPECT_EQ(false, r);
          if (r != false)
            continue;
          //std::cerr << "Unpadding Throughput = " << float(2*h*w*sizeof(float))/et3;
          //std::cerr << " GB/s\t";
        }

#if CHECK_RESULTS
        ASSERT_EQ(queue_->enqueueReadBuffer(d_dst, CL_TRUE, 0, sizeof(float)*h*(w-pad_w),
              dst_gpu), CL_SUCCESS);
        // compute golden
        // [h/t][t][w] to [h/t][w][t]
        cpu_aos_asta(src, dst, h-pad_h, w-pad_w, a);
        // [h/t][w][t] to [h/t][t][w]
        cpu_soa_asta(dst, src, (w-pad_w)*a, A, a);
        EXPECT_EQ(0, compare_output(dst_gpu, src, (h-pad_h)*(w-pad_w)));
#endif
      }
    }
//}

  /*cl_ulong et3 = 0;
  // Unpadding
  if (pad_h > 0){
    r = cl_unpadding((*queue_)(), d_dst(), h - pad_h, w - pad_w, h, &et3);
    // This may fail
    EXPECT_EQ(false, r);
    if (r != false)
      continue;
    std::cerr << "Unpadding Throughput = " << float(2*h*w*sizeof(float))/et3;
    std::cerr << " GB/s\t";
  }*/

#if BRUTE
    if(et < max_et) max_et = et;
    }
  }
  std::cerr << "" << h << "," << w << "\t";
  std::cerr << "Max_Throughput = " << float(h*w*2*sizeof(float)) / max_et;
  std::cerr << " GB/s\n";
#else
  std::cerr << "Padding_Throughput = " << float(2*h*w*sizeof(float)*N)/et2;
  std::cerr << " GB/s\t";
  std::cerr << "Transpose_Throughput = " << float(2*h*w*sizeof(float)*N)/et;
  std::cerr << " GB/s\t";
  std::cerr << "Unpadding_Throughput = " << float(2*h*w*sizeof(float)*N)/et3;
  std::cerr << " GB/s\t";
  std::cerr << "Throughput = " << float(2*h*w*sizeof(float)*N)/(et + et2 + et3);
  std::cerr << " GB/s\n";
#endif
  free(src);
  free(dst);
  free(dst_gpu);
  }
}

// testing 0100 transformation AaBb->ABab
TEST_F(libmarshal_cl_test, test_0100) {
  int bs[] = {256}; //{32};
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
  r = cl_transpose_0100((*queue_)(), d_dst(), A, a, B, b, NULL);
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
