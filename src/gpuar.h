#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <iostream>
#include "gpu.h"

/***************************************************************************
 *                                CONSTANTS
 ***************************************************************************/
#ifndef FALSE
#define FALSE 0
#endif

#ifndef TRUE
#define TRUE 1
#endif

// #define EOF_CHAR (511)
// #define CDF_DIM (EOF_CHAR + 2)
/***************************************************************************
 *                                  MACROS
 ***************************************************************************/
/* set bit x to 1 in probability_t.  Bit 0 is MSB */
// #define MASK_BIT(x) (probability_t)(1 << (PRECISION - (1 + (x))))

/* indices for a symbol's lower and upper cumulative probability ranges */
#define LOWER(c) (c)
#define UPPER(c) ((c) + 1)
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

typedef uint16_t probability_t; /* probability count type */
typedef uint16_t symb_t;               /* probability count type */
typedef probability_t *probability_ptr;

/* number of bits used to compute running code values */
#define PRECISION (8 * sizeof(probability_t))

/* 2 bits less than precision. keeps lower and upper bounds from crossing. */
// #define MAX_PROBABILITY (1 << (PRECISION - 2))
// #define READ_ELEMENT ulonglong2
// #define bitbuffer unsigned char

struct AdaptiveProbabilityRange
{

    /* probability ranges for each symbol: [ranges[LOWER(c)], ranges[UPPER(c)]) */
    probability_t *ranges;
    // probability_t cumulativeProb;   /* cumulative probability  of all ranges */
};

struct __align__(16) BitPointer
{
    uint8_t *fp;             // out ,in              /* file pointer used by stdio functions */
    uint8_t bitBuffer = 0; // cache           /* bits waiting to be read/written */
    uint8_t bitCount = 0;    // count  ,cached_bits            /* number of bits in bitBuffer */
    size_t in_ptr = 0;       //
    size_t in_sz = 0;
};

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */
    __host__ __device__ BitPointer createBitPointer(unsigned char *data);
    __host__ __device__ unsigned short getCompressedSize(const void *src);
    __host__ __device__ unsigned short getUncompressedSize(const void *src);
    __host__ __device__ void write_header(unsigned int f, void *dst, size_t bytes);
    __host__ __device__ uint32_t read_header(void const *src, uint32_t bytes);
    __host__ void initConstantRange(AdaptiveProbabilityRange *r,int end_of_char);
    __host__ __device__ uint16_t arCompress(const symb_t *symbols, const uint16_t size, uint8_t *outFile, probability_ptr r);
    __host__ __device__ uint16_t arDecompress(BitPointer bfpIn, const uint16_t size, symb_t *symb_dec, const probability_ptr cdf);
    void garCompressExecutor(const symb_t *source, size_t size, uint8_t *destination, uint32_t numBlocks, uint32_t *eachPacketBinSize);
    void garDecompressExecutor(const uint8_t *source, size_t size, symb_t *destination, uint32_t numBlocks);
#ifdef __cplusplus
}
#endif /* __cplusplus */

template <typename T>
__host__ __device__ void printdata(size_t readSize, T *deviceUncompressedData, bool isOngpu, int data_id)
{
    printf("%d:\n", data_id);
    T *print_data;
    if (isOngpu)
    {
        print_data = (T *)malloc(readSize * sizeof(T));
        cudaMemcpy(print_data, deviceUncompressedData, readSize * sizeof(T), cudaMemcpyDeviceToHost);
    }
    else
    {
        print_data = deviceUncompressedData;
    }
    for (size_t i = 0; i < readSize; i++)
    {
        printf("%x ", print_data[i]);
    }
    printf("\n");
    if (isOngpu)
    {
        delete print_data;
    }
}
