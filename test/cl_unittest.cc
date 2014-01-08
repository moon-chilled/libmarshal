#include "local_cl.hpp"
#include <gtest/gtest.h>
#include <cstdlib>
#include <string>
#include <vector>
#include <math.h>
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
  for (int t = 1; t <= 4096; t*=2) {
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
    const int N = 1;
    const int WARM_UP = 0;
    cl_ulong et = 0;
    for (int n = 0; n < N+WARM_UP; n++) {
      if (n == WARM_UP)
        et = 0;
      //bool r = cl_soa_asta_pttwac((*queue_)(), d_dst(), h, w, t);
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
    std::cerr << "Performance = " << float(2*h*w*sizeof(float)*N)/et;
    std::cerr << " GB/s\t";
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

#define MAX_MEM 12288 // Shared memory in Fermi and Kepler. Use 4096 for other devices.
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
        std::cerr << "Performance = " << float(h*w*2*sizeof(float)*N) / et;
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
  //int w = 20;
  for (int w = 3; w <= 768; w*=4)
  for (int t=1; t<=8; t+=1) {
    //int h = (100*100*130+t-1)/t*t;
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

TEST_F(libmarshal_cl_test, full) {
  // dataset size from http://www8.cs.umu.se/research/uminf/reports/2009/001/part1.pdf
  // figure 12
#define RANDOM 1
#if RANDOM
  // For skinny matrices
  //const int h_max = 1048576; const int h_min = 4096; //2^22 - 2^12
  //const int w_max = 64; const int w_min = 2;
  //const int h_max = 999999; const int h_min = 9999; //Tests in Catanzaro's paper
  //const int w_max = 31; const int w_min = 2;
  // For general matrices
  //const int h_max = 6144; const int h_min = 1024; //2^13 - 2^10
  //const int w_max = 4096; const int w_min = 1536;
  const int h_max = 19999; const int h_min = 1000; //Tests in Catanzaro's paper
  const int w_max = 19999; const int w_min = 1000;
  //const int h_max = 9999; const int h_min = 1000;
  //const int w_max = 9999; const int w_min = 1000;

  srand(time(NULL));
  for (int n = 0; n < 10; n++){
  // Generate random dimensions
  //srand(n*time(NULL));
  int h = rand() % (h_max-h_min+1) + h_min;
  //srand(n*2*time(NULL));
  int w = rand() % (w_max-w_min+1) + w_min;
#else 
  //int ws[] = {1800, 2500, 3200, 3900, 5100, 7200}; //Matrix sizes in PPoPP2014 paper
  //int hs[] = {7200, 5100, 4000, 3300, 2500, 1800};
  int ws[] = {9703, 9789, 17084, 19193, 7391, 17853, 15257, 10148, 18801, 12624, 12918, 14858};
  int hs[] = {19604, 12903, 2299, 1796, 6024, 18201, 7844, 9189, 12050, 10645, 7244, 11588};
  for (int n = 0; n < 12; n++) {
  int w = ws[n];
  int h = hs[n];
#endif
  std::cerr << "" << h << "," << w << "\t";

  float *src = (float*)malloc(sizeof(float)*h*w);
  float *dst = (float*)malloc(sizeof(float)*h*w);
  float *dst_gpu = (float*)malloc(sizeof(float)*h*w);
  generate_vector(src, h*w);

  Factorize hf(h), wf(w);
  hf.tiling_options();
  wf.tiling_options();
  std::vector<int> hoptions = hf.get_tile_sizes();
  std::vector<int> woptions = wf.get_tile_sizes();
  std::cerr << "" << hoptions.size() << "," << woptions.size() << "\t";

#if 1
  // Sort factors
  //for(int x=0; x<hoptions.size(); x++) printf("%d ", hoptions[x]);
  //printf("\n");
  size_t hf_sorted[hoptions.size()];
  size_t wf_sorted[woptions.size()];
  gsl_sort_int_index((size_t *)hf_sorted, &hoptions[0], 1, hoptions.size());
  for(int x=0; x<hoptions.size(); x++) printf("%d ", hoptions[hf_sorted[x]]);
  printf("\n");
  gsl_sort_int_index((size_t *)wf_sorted, &woptions[0], 1, woptions.size());
  for(int x=0; x<woptions.size(); x++) printf("%d ", woptions[wf_sorted[x]]);
  printf("\n");
#endif

#define BRUTE 0
#if BRUTE
  for (int i = 0 ; i < hoptions.size(); i++) {
  //for (int i = 0 ; i < 1; i++) {
    //int A = h/hoptions[i], a = hoptions[i];
    int A = h/hoptions[hf_sorted[i]], a = hoptions[hf_sorted[i]];
    for (int j = 0; j < woptions.size(); j++) {
    //for (int j = 0; j < 1; j++) {
      //int B = w/woptions[j], b = woptions[j];
      int B = w/woptions[wf_sorted[j]], b = woptions[wf_sorted[j]];
      if (B > 5900) continue;
#else
  // Heuristic used by Catanzaro et al. (PPoPP'2014)
  /*int A, a, B, b;
  int i = 0;
  if (hoptions.size() == 1){
    a = 1; A = h;
  }
  else{
    while (hoptions[hf_sorted[i]] <=72){ //110
      i++;
    }
    a = hoptions[hf_sorted[i-1]]; A = h/a;
  }
  i = 0;
  if (woptions.size() == 1){
    b = 1; B = w;
  }
  else{
    while (woptions[wf_sorted[i]] <=72){ //110
      i++;
    }
    b = woptions[wf_sorted[i-1]]; B = w/b;
  }*/

#if 0
  // Heu 2_3
  int A, a, B, b;
  int i = 0;
  if (hoptions.size() == 1){
    if (hoptions[hf_sorted[0]] <= 6112){
      a = h; A = 1;
      i = 0;
      for (int j = 0; j < woptions.size(); j++)
        if (((woptions[wf_sorted[j]]*a+31)/32) + ((((woptions[wf_sorted[j]]*a+31)/32)>>5)*1) <= 12288)
          i++;
      if (i > 0){
        b = woptions[wf_sorted[i-1]]; B = w/b;
      }
      else{
        b = 1; B = w/b;
      } 
    }
    else{
      a = 1; A = h;
      i = 0;
      for (int j = 0; j < woptions.size(); j++)
        if (woptions[wf_sorted[j]] * a < 12288)
          i++;
      if (i > 0){
        b = woptions[wf_sorted[i-1]]; B = w/b;
      }
      else{
        b = 1; B = w/b;
      } 
    }
  }
  else if (woptions.size() == 1){
    if (woptions[wf_sorted[0]] <= 6112){
      b = w; B = 1;
      i = 0;
      for (int j = 0; j < hoptions.size(); j++)
        if (((hoptions[hf_sorted[j]]*b+31)/32) + ((((hoptions[hf_sorted[j]]*b+31)/32)>>5)*1) <= 12288)
          i++;
      if (i > 0){
        a = hoptions[hf_sorted[i-1]]; A = h/a;
      }
      else{
        a = 1; A = h/b;
      } 
    }
    else{
      b = 1; B = w;
      i = 0;
      for (int j = 0; j < hoptions.size(); j++)
        if (hoptions[hf_sorted[j]] * b < 12288)
          i++;
      if (i > 0){
        a = hoptions[hf_sorted[i-1]]; A = h/a;
      }
      else{
        a = 1; A = h/b;
      } 
    }
  }
  else{
    /*//i = 0;
    int k = 0;
    for (int j = 0; j < hoptions.size(); j++)
      //if (hoptions[hf_sorted[i]] <= 128)
      //  i++;
      if (hoptions[hf_sorted[k]] <= 128)
        k++;
    //if (i > 0){
    //  a = hoptions[hf_sorted[i-1]]; A = h/a;
    if (k > 0){
      a = hoptions[hf_sorted[k-1]]; A = h/a;
    }
    else{
      a = hoptions[hf_sorted[0]]; A = h/a;
    }
    i = 0;
    for (int j = 0; j < woptions.size(); j++)
      if (woptions[wf_sorted[j]] * a < 12288 && woptions[wf_sorted[j]] <= 128) //6144, 192
        i++;
    b = woptions[wf_sorted[i-1]]; B = w/b;
    if (b * A < 12288 && b * A > b * a){ //6144
      A = a; a = h/A;
      for (int j = k-2; j >= 0; j--)
        if (b * h/hoptions[hf_sorted[j]] < 6144){
          a = h/hoptions[hf_sorted[j]]; A = h/a;
        }
    }*/
    // Heu 4
    int k = 0;
    for (int j = 0; j < woptions.size(); j++)
      if (woptions[wf_sorted[k]] < 32)
        k++;
    if (k > 0){
      b = woptions[wf_sorted[k]]; B = w/b;
    }
    else{
      b = woptions[wf_sorted[0]]; B = w/b;
    }
    i = 0;
    for (int j = 0; j < hoptions.size(); j++)
      if (hoptions[hf_sorted[j]] * b < 12288)
        i++;
    if (i > 0){
      a = hoptions[hf_sorted[i-1]]; A = h/a;
    }
    else{
      a = hoptions[hf_sorted[0]]; A = h/a;
    }
  }
#endif

  int A = 0; int a = 0; int B = 0; int b = 0;
  struct int2{int x; int y;};
  int k = 0; int l = 0; int p = 0; 
  int2 maxtile; maxtile.x = 0; maxtile.y = 0;
  int re = 0; int min_limit = 24; int done = 0; int max_limit = 4096;
  int hoptions_good[hoptions.size()];
  int woptions_good[woptions.size()];
  int2 tileoptions[hoptions.size()*woptions.size()];
do{
  printf("1. done = %d\t", done);
  k = 0; l = 0; p = 0; re = 0;
  // Desired minimum and maximum for a and b
//if (k == 0)
  for (int j = 0; j < hoptions.size(); j++)
    if (hoptions[hf_sorted[j]] >= min_limit && hoptions[hf_sorted[j]] <= max_limit){
      hoptions_good[k] = hoptions[hf_sorted[j]];
      k++;
    }
//if (l == 0)
  for (int j = 0; j < woptions.size(); j++)
    if (woptions[wf_sorted[j]] >= min_limit && woptions[wf_sorted[j]] <= max_limit){
      woptions_good[l] = woptions[wf_sorted[j]];
      l++;
    }
  printf("k=%d, l=%d\t",k,l);
  //if (k == 0 && hoptions.size() > 1) k = hoptions.size();
  //if (l == 0 && woptions.size() > 1) l = woptions.size();
  // Two in the desired range
  if (k > 0 && l > 0){
    for (int i = 0; i < k; i++)
      for (int j = 0; j < l; j++)
        if (hoptions_good[i] * woptions_good[j] < 12288){ // Fits in local memory
          tileoptions[p].x = hoptions_good[i]; 
          tileoptions[p].y = woptions_good[j];
          p++;
        }
    //printf("p=%d\t",p);
    int maxfactor = 1;
    for (int j = 0; j < p; j++){ // Use as much local memory as possible
      done = 1;
      int tilesize = tileoptions[j].x * tileoptions[j].y;
      int factor = 1;
      //printf("tilesize = %d\t", tilesize);
      if (tilesize < 768) factor = 16;
      else if (tilesize >= 768 && tilesize < 1536) factor = 8;
      else if (tilesize >= 1536 && tilesize < 3072) factor = 4;
      else if (tilesize >= 3072 && tilesize < 6144) factor = 2;
      tilesize *= factor;
      //printf("tilesize = %d\n", tilesize);
      if (tilesize > maxtile.x * maxtile.y * maxfactor){
        maxtile.x = tileoptions[j].x;
        maxtile.y = tileoptions[j].y;
        maxfactor = factor;
      }
    }
    if (p == 0 && min_limit <= 0) // && max_limit > 6144) // Does not fit in local memory: largest a and b possible
      for (int i = 0; i < k; i++)
        for (int j = re; j < l; j++){
          //if (((hoptions_good[i]*woptions_good[j]+31)/32) + ((((hoptions_good[i]*woptions_good[j]+31)/32)>>5)*1) <= 12288 && j >= re){
          //printf("re=%d\t%d\t",re, ((hoptions_good[i]*woptions_good[j]+31)/32) + ((((hoptions_good[i]*woptions_good[j]+31)/32)>>5)*1));
          if (((hoptions_good[i]*woptions_good[j]+31)/32) + ((((hoptions_good[i]*woptions_good[j]+31)/32)>>5)*1) <= 12288){
            maxtile.x = hoptions_good[i];
            maxtile.y = woptions_good[j];
            re = j;
            done = 1;
          }}
    //if (p == 0 && min_limit >= 8 && max_limit <= 6144){
    if (p == 0){
      min_limit -= 2;
      max_limit += 256;
    //printf("in min_limit=%d, max_limit=%d\t",min_limit, max_limit);
    }
    if (done == 1){
      a = maxtile.x; A = h/a;
      b = maxtile.y; B = w/b;
      //printf("a = %d, b = %d\n", a, b);
    }
  }
  else{
    min_limit -= 2;
    max_limit += 256;
    //printf("out min_limit=%d, max_limit=%d\t",min_limit, max_limit);
  }
  // One in the desired range
  /*if (k == 0 && l > 0){
    for (int i = 0; i < hoptions.size(); i++)
      for (int j = 0; j < l; j++)
        if (hoptions_good[i] * woptions_good[j] < 12288){ // Fits in local memory
          tileoptions[p].x = hoptions_good[i];
          tileoptions[p].y = woptions_good[j];
          p++;
        }
    int maxfactor = 1;
    for (int j = 0; j < p; j++){ // Use as much local memory as possible
      done = 1;
      int tilesize = tileoptions[j].x * tileoptions[j].y;
      int factor = 1;
      if (tilesize < 768) factor = 16;
      else if (tilesize >= 768 && tilesize < 1536) factor = 8;
      else if (tilesize >= 1536 && tilesize < 3072) factor = 4;
      else if (tilesize >= 3072 && tilesize < 6144) factor = 2;
      tilesize *= factor;
      if (tilesize > maxtile.x * maxtile.y * maxfactor){
        maxtile.x = tileoptions[j].x;
        maxtile.y = tileoptions[j].y;
        maxfactor = factor;
      }
    }
    if (p == 0) // Does not fit in local memory: largest a and b possible
      for (int i = 0; i < hoptions.size(); i++)
        for (int j = re; j < l; j++)
          if (((hoptions_good[i]*woptions_good[j]+31)/32) + ((((hoptions_good[i]*woptions_good[j]+31)/32)>>5)*1) <= 12288){
            maxtile.x = hoptions_good[i];
            maxtile.y = woptions_good[j];
            re = j;
            done = 1;
          }
    if (done == 0) min_limit -= 24;
    else{
      a = maxtile.x; A = h/a;
      b = maxtile.y; B = w/b;
    }
  }
  if (k > 0 && l == 0){
    for (int i = 0; i < woptions.size(); i++)
      for (int j = 0; j < k; j++)
        if (hoptions_good[j] * woptions_good[i] < 12288){ // Fits in local memory
          tileoptions[p].x = hoptions_good[j];
          tileoptions[p].y = woptions_good[i];
          p++;
        }
    int maxfactor = 1;
    for (int j = 0; j < p; j++){ // Use as much local memory as possible
      done = 1;
      int tilesize = tileoptions[j].x * tileoptions[j].y;
      int factor = 1;
      if (tilesize < 768) factor = 16;
      else if (tilesize >= 768 && tilesize < 1536) factor = 8;
      else if (tilesize >= 1536 && tilesize < 3072) factor = 4;
      else if (tilesize >= 3072 && tilesize < 6144) factor = 2;
      tilesize *= factor;
      if (tilesize > maxtile.x * maxtile.y * maxfactor){
        maxtile.x = tileoptions[j].x;
        maxtile.y = tileoptions[j].y;
        maxfactor = factor;
      }
    }
    if (p == 0) // Does not fit in local memory: largest a and b possible
      for (int i = 0; i < woptions.size(); i++)
        for (int j = re; j < k; j++)
          if (((hoptions_good[j]*woptions_good[i]+31)/32) + ((((hoptions_good[j]*woptions_good[i]+31)/32)>>5)*1) <= 12288){
            maxtile.x = hoptions_good[j];
            maxtile.y = woptions_good[i];
            re = j;
            done = 1;
          }
    if (done == 0) min_limit -= 24;
    else{
      a = maxtile.x; A = h/a;
      b = maxtile.y; B = w/b;
    }
  }*/
}while(!done && min_limit >= 0);
  k = 0; l = 0;
  // None in the desired range
  //if (k == 0 && l == 0){
  if (done == 0){
    //done = 1;
    printf("2. done = %d\t", done);
    for (int j = 0; j < hoptions.size(); j++)
      if (hoptions[hf_sorted[j]] <= 6112)
        k++;
    for (int j = 0; j < woptions.size(); j++)
      if (woptions[wf_sorted[j]] <= 6112)
        l++;
    if (k > 0){
      a = hoptions[hf_sorted[k]]; A = h/a;
    }
    else{
      a = 1; A = h/a;
    }
    if (l > 0){
      b = woptions[hf_sorted[l]]; B = w/b;
    }
    else{
      b = 1; B = w/b;
   }
  }
#endif
      //if (B > 5900) continue;

      cl_int err;
      cl::Buffer d_dst = cl::Buffer(*context_, CL_MEM_READ_WRITE,
          sizeof(float)*h*w, NULL, &err);
      ASSERT_EQ(err, CL_SUCCESS);
      err = queue_->enqueueWriteBuffer(
            d_dst, CL_TRUE, 0, sizeof(float)*h*w, src);
      EXPECT_EQ(err, CL_SUCCESS);
      if (err != CL_SUCCESS)
        continue;
      std::cerr << "" << A << "," << a << ",";
      std::cerr << "" << B << "," << b <<",";

      bool r = false;
      //r = cl_transpose((*queue_)(), d_dst(), A, a, B, b);
      // 1 = Spreading factor, change if needed - JGL
      //r = cl_transpose((*queue_)(), d_dst(), A, a, B, b, 1, NULL); 

      // Calculate spreading factor
      //int S_f = (MAX_MEM-512) / (((a*b+31)/32) + ((((a*b+31)/32)>>5)*P));
      int S_f = MAX_MEM / (((a*b+31)/32) + ((((a*b+31)/32)>>5)*P));
      std::cerr << "S_f = " << S_f << ",";
      if (S_f < 2) S_f = 1;
      else if (S_f >=2 && S_f < 4) S_f = 2;
      else if (S_f >=4 && S_f < 8) S_f = 4;
      else if (S_f >=8 && S_f < 16) S_f = 8;
      else if (S_f >=16 && S_f < 32) S_f = 8; //16;
      else S_f = 32; // BS will be used
      std::cerr << "" << S_f << ",\t";

      r = cl_transpose((*queue_)(), d_dst(), A, a, B, b, S_f, NULL); 

      // This may fail
      EXPECT_EQ(false, r);
      if (r != false)
        continue;
      // compute golden
      // [h/t][t][w] to [h/t][w][t]
//      cpu_aos_asta(src, dst, h, w, a);
      // [h/t][w][t] to [h/t][t][w]
//      cpu_soa_asta(dst, src, w*a, A, a);
      ASSERT_EQ(queue_->enqueueReadBuffer(d_dst, CL_TRUE, 0, sizeof(float)*h*w,
            dst_gpu), CL_SUCCESS);
//      EXPECT_EQ(0, compare_output(dst_gpu, src, h*w));
#if BRUTE
    }
  }
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
