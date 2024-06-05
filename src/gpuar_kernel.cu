/***************************************************************************
 *                 Arithmetic Encoding and Decoding Library
 *
 *   Purpose : Use arithmetic coding to compress/decompress streams
 *   Original author for host code implementation: https://github.com/fab-jul/torchac/blob/master/torchac/backend/torchac_backend.cpp
 *   Optimize to CUDA implementation: Chunyang Fu.
 *   Date    : March 19, 2023
 *
 *****************************************************************************/
#include "gpu.h"
#include "gpuar.h"
#include "assert.h"

__managed__ int EOF_CHAR;
__managed__ int CDF_DIM;
__constant__ AdaptiveProbabilityRange INITIALIZED_RANGE;
__constant__ probability_t INITIALIZED_CUMULATIVE_PROB;

//__________________________________torch ac file ___________________________________

__host__ __device__ BitPointer createBitPointer(uint8_t *data)
{
    BitPointer bp;
    bp.fp = data;     /* file pointer used by stdio functions */
    bp.bitBuffer = 0; /* bits waiting to be read/written */
    bp.bitCount = 0;  /* number of bits in bitBuffer */
    bp.in_sz = 0;
    bp.in_ptr = 0;
    return bp;
}
__host__ __device__ int32_t putChar(const int32_t c, BitPointer *stream)
{
    stream->fp[0] = c;
    ++stream->fp;
    return c;
}

__host__ __device__ void append(const int bit, BitPointer *stream)
{
    stream->bitCount += 1;
    stream->bitBuffer <<= 1;
    stream->bitBuffer |= bit;
    if (stream->bitCount == 8)
    {
        putChar(stream->bitBuffer, stream);
        stream->bitBuffer = 0;
        stream->bitCount = 0;
    }
}

__host__ __device__ void flush(BitPointer *stream)
{
    if (stream->bitCount > 0)
    {
        for (int i = stream->bitCount; i < 8; ++i)
        {
            append(0, stream);
        }
        assert(stream->bitCount == 0);
    }
}

__host__ __device__ void append_bit_and_pending(const int bit, uint64_t &pending_bits, BitPointer *stream)
{
    append(bit, stream);
    while (pending_bits > 0)
    {
        append(!bit, stream);
        pending_bits -= 1;
    }
}

//__________________________________torch ac file ___________________________________

__host__ __device__ uint32_t read_header(void const *src, uint32_t bytes)
{

    uint8_t *p = (uint8_t *)src;
    switch (bytes)
    {
    case 4:
        return (*p | *(p + 1) << 8 | *(p + 2) << 16 | *(p + 3) << 24);
    case 3:
        return (*p | *(p + 1) << 8 | *(p + 2) << 16);
    case 2:
        return (*p | *(p + 1) << 8);
    case 1:
        return (*p);
    }
    return 0;
}

__host__ __device__ uint16_t getCompressedSize(const void *src)
{

    return read_header(((const int8_t *)src), 2);
}

__host__ __device__ uint16_t getUncompressedSize(const void *src)
{

    return read_header(((const int8_t *)src) + 2, 2);
}

__host__ __device__ void write_header(uint32_t f, void *dst, size_t bytes)
{

    uint8_t *p = (uint8_t *)dst;

    switch (bytes)
    {
    case 4:
        *p = (uint8_t)f;
        *(p + 1) = (uint8_t)(f >> 8);
        *(p + 2) = (uint8_t)(f >> 16);
        *(p + 3) = (uint8_t)(f >> 24);
        return;
    case 3:
        *p = (uint8_t)f;
        *(p + 1) = (uint8_t)(f >> 8);
        *(p + 2) = (uint8_t)(f >> 16);
        return;
    case 2:
        *p = (uint8_t)f;
        *(p + 1) = (uint8_t)(f >> 8);
        return;
    case 1:
        *p = (uint8_t)f;
        return;
    }
}


__host__ void initConstantRange(AdaptiveProbabilityRange *r,int end_of_char)
{
    cudaMemcpyToSymbol(INITIALIZED_RANGE, r, sizeof(AdaptiveProbabilityRange));
    cudaMemcpyToSymbol(INITIALIZED_CUMULATIVE_PROB, r->ranges + UPPER(EOF_CHAR), sizeof(probability_t));
    checkCudaErrors(cudaMemcpyToSymbol(EOF_CHAR,&end_of_char,sizeof(int)));
    int cdf_dim = end_of_char +2;
    cudaMemcpyToSymbol(CDF_DIM,&cdf_dim,sizeof(int));
}

/***************************************************************************
 *   Function   : arCompress
 *   Description: This routine generates a list of arithmetic code ranges for
 *                a file and then uses them to write_gpu out an encoded version
 *                of that file.
 *   Parameters : inFile - Pointer of stream to encode
 *                outFile - Pointer of stream to write_gpu encoded output to
 *   Effects    : Binary data is arithmetically encoded
 *   Returned   : TRUE for success, otherwise FALSE.
 ***************************************************************************/
__host__ __device__ uint16_t arCompress(const symb_t *symbols, const uint16_t size, uint8_t *outFile, const probability_ptr cdf)
{
    BitPointer out_cache = createBitPointer(outFile + PACKET_HEADER_LENGTH);
    uint32_t low = 0;
    uint32_t high = 0xFFFFFFFFU;
    uint64_t pending_bits = 0;
    const int precision = 16;
    // int endofchar = EOF_CHAR;
    // // cudaMemcpyFromSymbol(&endofchar, EOF_CHAR, sizeof(int));
    // const int cdf_dim = endofchar+2;

    for (int i = 0; i < size; ++i)
    {
        const int16_t sym_i = symbols[i];
        // printf("symb %d EOF_CHAR %d\n ",sym_i,endofchar);
        const uint64_t span = static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1;
        // const int z = EOF_CHAR;
        const int offset = i * CDF_DIM;
        // printf("offset %d",offset);
        // Left boundary is at offset + sym_i
        const uint32_t c_low = cdf[offset + sym_i];
        // Right boundary is at offset + sym_i + 1, except for the `EOF_CHAR`
        const uint32_t c_high = sym_i == EOF_CHAR ? 0x10000U : cdf[offset + sym_i + 1];

        high = (low - 1) + ((span * static_cast<uint64_t>(c_high)) >> precision);
        low = (low) + ((span * static_cast<uint64_t>(c_low)) >> precision);

        while (true)
        {
            if (high < 0x80000000U)
            {
                append_bit_and_pending(0, pending_bits, &out_cache);
                low <<= 1;
                high <<= 1;
                high |= 1;
            }
            else if (low >= 0x80000000U)
            {
                append_bit_and_pending(1, pending_bits, &out_cache);
                low <<= 1;
                high <<= 1;
                high |= 1;
            }
            else if (low >= 0x40000000U && high < 0xC0000000U)
            {
                pending_bits++;
                low <<= 1;
                low &= 0x7FFFFFFF;
                high <<= 1;
                high |= 0x80000001;
            }
            else
            {
                break;
            }
        }
    }

    pending_bits += 1;

    if (pending_bits)
    {
        if (low < 0x40000000U)
        {
            append_bit_and_pending(0, pending_bits, &out_cache);
        }
        else
        {
            append_bit_and_pending(1, pending_bits, &out_cache);
        }
    }
    flush(&out_cache);

    uint32_t length = out_cache.fp - outFile;
    // printf("length %d , size %d\n", length,size );
    write_header(length, outFile, 2);
    // write_header(size, outFile + 2, 2);
    assert(length<=COMPRESSED_PACKET_SIZE);
    return length;
}

__host__ __device__ int32_t getChar(BitPointer *stream)
{
    //;

    int32_t x = stream->fp[0];
    ++stream->fp;

    return x;
}


__host__ __device__ void readBinBits(BitPointer *bfpIn, uint32_t &value)
{
    if (bfpIn->bitCount == 0)
    {
        if (bfpIn->in_ptr == bfpIn->in_sz)
        {
            value <<= 1;
            return;
        }
        /// Read 1 byte
        bfpIn->bitBuffer = (uint8_t)bfpIn->fp[bfpIn->in_ptr];
        bfpIn->in_ptr++;
        bfpIn->bitCount = 8;
    }
    value <<= 1;
    value |= (bfpIn->bitBuffer >> (bfpIn->bitCount - 1)) & 1;
    bfpIn->bitCount--;
}

__host__ __device__ void decoderInitialize(BitPointer *bfpIn, uint32_t &value)
{
    for (int i = 0; i < 32; ++i)
    {
        readBinBits(bfpIn, value);
    }
}

__host__ __device__ probability_t binsearch(const probability_t *cdf, probability_t target, probability_t max_sym,
                                            const int offset) /* i * Lp */
{
    probability_t left = 0;
    probability_t right = max_sym + 1; // len(cdf) == max_sym + 2
    while (left + 1 < right)
    {
        const probability_t m = (left + right) / 2;//static_cast<const probability_t>((left + right) / 2);
        const auto v = cdf[offset + m];
        if (v < target)
        {
            left = m;
        }
        else if (v > target)
        {
            right = m;
        }
        else
        {
            return m;
        }
    }
    return left;
}
/***************************************************************************
 *   Function   : arDecompress
 *   Description: This routine opens an arithmetically encoded file, reads
 *                it's header, and builds a list of probability ranges which
 *                it then uses to decode the rest of the file.
 *   Parameters : inFile - Pointer to stream to decode
 *                outFile - Pointer to stream to write_gpu decoded output to
 *   Effects    : Encoded file is decoded
 *   Returned   : TRUE for success, otherwise FALSE.
 ***************************************************************************/
__host__ __device__ uint16_t arDecompress(BitPointer bfpIn, const uint16_t size, symb_t *symb_dec, const probability_ptr cdf)
{
    uint32_t low = 0;
    uint32_t high = 0xFFFFFFFFU;
    uint32_t value = 0;
    const uint32_t c_count = 0x10000U;
    const int precision = 16;

    // InCacheString in_cache(in);

    // bfpIn.
    // de initialize(value);
    decoderInitialize(&bfpIn, value);

    for (int i = 0; i < size; ++i)
    {
        const uint64_t span = static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1;
        // always < 0x10000 ???
        const uint16_t count = ((static_cast<uint64_t>(value) - static_cast<uint64_t>(low) + 1) * c_count - 1) / span;

        const int offset = i * CDF_DIM;
        auto sym_i = binsearch(cdf, count, (probability_t)EOF_CHAR, offset);

        symb_dec[i] = (int16_t)sym_i;
        // printf("739 %d\n",symb_dec[i]);
        if (i == size - 1)
        {
            break;
        }

        const uint32_t c_low = cdf[offset + sym_i];
        const uint32_t c_high = sym_i == EOF_CHAR ? 0x10000U : cdf[offset + sym_i + 1];

        high = (low - 1) + ((span * static_cast<uint64_t>(c_high)) >> precision);
        low = (low) + ((span * static_cast<uint64_t>(c_low)) >> precision);

        while (true)
        {
            if (low >= 0x80000000U || high < 0x80000000U)
            {
                low <<= 1;
                high <<= 1;
                high |= 1;
                readBinBits(&bfpIn, value);
            }
            else if (low >= 0x40000000U && high < 0xC0000000U)
            {
                /**
                 * 0100 0000 ... <= value <  1100 0000 ...
                 * <=>
                 * 0100 0000 ... <= value <= 1011 1111 ...
                 * <=>
                 * value starts with 01 or 10.
                 * 01 - 01 == 00  |  10 - 01 == 01
                 * i.e., with shifts
                 * 01A -> 0A  or  10A -> 1A, i.e., discard 2SB as it's all the same while we are in
                 *    near convergence
                 */
                low <<= 1;
                low &= 0x7FFFFFFFU; // make MSB 0
                high <<= 1;
                high |= 0x80000001U; // add 1 at the end, retain MSB = 1
                value -= 0x40000000U;
                readBinBits(&bfpIn, value);
            }
            else
            {
                break;
            }
        }
    }
    return 0;
}

__global__ void garCompress(const symb_t *source, size_t size, uint8_t *destination, uint32_t *eachPacketBinSize)
{
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x; // 0~31
    const uint32_t startPosition = index * UNCOMPRESSED_PACKET_SIZE;
    if (startPosition < size)
    {
        probability_t *cdf = INITIALIZED_RANGE.ranges + ((index * UNCOMPRESSED_PACKET_SIZE) * CDF_DIM);
        size_t packetSize = MIN(size - startPosition, UNCOMPRESSED_PACKET_SIZE);
        eachPacketBinSize[index] = arCompress(source + startPosition, packetSize, destination + (index * COMPRESSED_PACKET_SIZE), cdf);
    }
}

__global__ void garDecompress(const uint8_t *source, size_t size, symb_t *symb_dec)
{
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x; // 0~31
    const uint32_t startPosition = index * UNCOMPRESSED_PACKET_SIZE;
    const uint8_t *startSrc = source + (index * COMPRESSED_PACKET_SIZE);
    symb_t *data_dec = symb_dec + (index * UNCOMPRESSED_PACKET_SIZE);

    if (startPosition < size)
    {
        probability_t *cdf = INITIALIZED_RANGE.ranges + ((index * UNCOMPRESSED_PACKET_SIZE) * (EOF_CHAR + 2));
        size_t unz = MIN(size - startPosition, UNCOMPRESSED_PACKET_SIZE);
        BitPointer bfpIn = createBitPointer((uint8_t *)startSrc + PACKET_HEADER_LENGTH);
        bfpIn.in_sz = getCompressedSize(startSrc);
        // printf("index %d unz %ld  size %ld\n ",index,unz,size);
        arDecompress(bfpIn, unz, data_dec, cdf);
    }
}

void garDecompressExecutor(const uint8_t *source, size_t size, symb_t *destination, uint32_t numBlocks)
{
    garDecompress<<<numBlocks, NUM_THREADS>>>(source, size, destination);
}

void garCompressExecutor(const symb_t *source, size_t size, uint8_t *destination, uint32_t numBlocks, uint32_t *eachPacketBinSize)
{
    garCompress<<<numBlocks, NUM_THREADS>>>(source, size, destination, eachPacketBinSize);
}


