#ifndef _LIBMARSHAL_INCLUDE_MARSHAL_H_
#define _LIBMARSHAL_INCLUDE_MARSHAL_H_

extern "C" {

bool gpu_aos_asta_bs(float *src, int height, int width, int tile_size, clock_t *timer);

bool gpu_aos_asta_pttwac(float *src, int height, int width, int tile_size, clock_t *timer);

bool gpu_soa_asta_pttwac(float *src, int height, int width, int tile_size, clock_t *timer);
};

#endif // _LIBMARSHAL_INCLUDE_MARSHAL_H_
