CUDA_ADD_EXECUTABLE(clk clock.cu)
CUDA_ADD_EXECUTABLE(unittest unittest.cc)
TARGET_LINK_LIBRARIES(unittest gtest gtest_main)
