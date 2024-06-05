#pragma once

#include "compressor.h"

#include "progress_monitor.h"
namespace gip
{
    class GPUCompressor : public Compressor
    {
    private:
        constexpr static int NUM_STREAMS = 1;

        uint32_t numBlocks;
        uint32_t totalThreads;
        size_t uncompressedDataBufferSize;
        cudaStream_t inputStreams[NUM_STREAMS]; //(cudaStream_t*) malloc(nstreams * sizeof(cudaStream_t));
        cudaStream_t outputStream;              //(cudaStream_t*) malloc(nstreams * sizeof(cudaStream_t));

        symb_t *devieSymbolData = NULL;
        uint8_t *deviceBinData = NULL;
        char *inputPacketBuffer = NULL;
        uint8_t *outputPacketBuffer = NULL;
        uint32_t *eachPacketBinSize = NULL;
        // uint32_t *eachPacketBinSize =  NULL;

    public:
        CompressionInfo compress(ProgressMonitor *monitor, symb_t *symbols, size_t size);
        CompressionInfo decompress(ProgressMonitor *monitor, symb_t *symbols_dec, size_t size);
        GPUCompressor(probability_t *data, int &deviceID, size_t end_of_char);
        ~GPUCompressor();

        void chooseDevice(const int id);

        static unsigned short getPacketSize(const uint8_t *packet)
        {
            return ((unsigned short *)(((char *)packet)))[0];
        }

        void allocateResource();
        void cleanResource();
    };
}