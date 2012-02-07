CUDA_ADD_EXECUTABLE(clk clock.cu)
CUDA_ADD_EXECUTABLE(unittest unittest.cu marshal.cu)
TARGET_LINK_LIBRARIES(unittest gtest gtest_main)
