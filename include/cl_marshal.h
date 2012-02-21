//===--- marshal.h - GPU in-place marshaling library          ----------===//
// (C) Copyright 2012 The Board of Trustees of the University of Illinois.
// All rights reserved.
//
//                            libmarshal
// Developed by:
//                           IMPACT Research Group
//                  University of Illinois, Urbana-Champaign
// 
// This file is distributed under the Illinois Open Source License.
// See LICENSE.TXT for details.
//
// Author: I-Jui Sung (sung10@illinois.edu)
//
//===---------------------------------------------------------------------===//
//
//  This file declares the interface of the libmarshal 
//
//===---------------------------------------------------------------------===//

#ifndef _LIBMARSHAL_INCLUDE_MARSHAL_H_
#define _LIBMARSHAL_INCLUDE_MARSHAL_H_
#include <cl.h>
extern "C" {

bool cl_aos_asta_bs(cl_command_queue queue,
    cl_mem src, int height, int width, int tile_size);
bool cl_aos_asta_pttwac(cl_command_queue queue, cl_mem src, int height,
    int width, int tile_size);
bool cl_soa_asta_pttwac(cl_command_queue queue, cl_mem src, int height,
    int width, int tile_size);

}

#endif // _LIBMARSHAL_INCLUDE_MARSHAL_H_
