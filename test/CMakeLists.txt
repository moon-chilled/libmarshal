CUDA_ADD_EXECUTABLE(clk clock.cu)
set(CUDA_NVCC_FLAGS -arch sm_12)
CUDA_ADD_EXECUTABLE(unittest unittest.cu marshal.cu)
TARGET_LINK_LIBRARIES(unittest gtest gtest_main)
