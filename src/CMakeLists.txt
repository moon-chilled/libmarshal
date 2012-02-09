set(CUDA_NVCC_FLAGS -arch sm_12)
CUDA_ADD_LIBRARY(marshal marshal.cu)
