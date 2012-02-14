IF (CUDA_FOUND)
SET (CUDA_NVCC_FLAGS -arch sm_12)
CUDA_ADD_LIBRARY(marshal marshal.cu)
ENDIF (CUDA_FOUND)

ADD_CUSTOM_COMMAND(
    OUTPUT embd.cpp embd.hpp
    COMMAND embd ${CMAKE_CURRENT_SOURCE_DIR}/cl/cl_aos_asta.cl @ cl/cl_aos_asta.cl
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} 
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cl/cl_aos_asta.cl
    )
INCLUDE_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR}) # for embd.hpp
ADD_LIBRARY(cl_marshal cl_marshal.cc embd.cpp)
