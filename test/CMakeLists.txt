CUDA_ADD_EXECUTABLE(clk clock.cu)
set(CUDA_NVCC_FLAGS -arch sm_12)
CUDA_ADD_EXECUTABLE(unittest unittest.cu)
TARGET_LINK_LIBRARIES(unittest marshal gtest gtest_main)
