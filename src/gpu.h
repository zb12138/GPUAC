#pragma once

#include <cuda_runtime.h>
#include <helper_timer.h>
#include <helper_cuda.h>
#include <iostream>
#include <cassert>
#define NUM_THREADS 32
#define PROCESSORS_GRIDS_FACTOR 8
#define UNCOMPRESSED_PACKET_SIZE (8192) // the number of symbols in a packet
#define COMPRESSED_PACKET_SIZE (UNCOMPRESSED_PACKET_SIZE*2+1+PACKET_HEADER_LENGTH) // Need 2 bytes to encode a symbol in worst-case.
#define PACKET_HEADER_LENGTH 2 // the header of a packet [length of bin size; number of symbols in the bin]
#define cutilCheckError

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

#ifdef __cplusplus
}
#endif /* __cplusplus */