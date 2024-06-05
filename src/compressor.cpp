#include "compressor.h"

using namespace gip;

gip::Compressor::Compressor() : openFileName(""), saveFileName(""), process_timer(nullptr), io_timer(nullptr),
                                openFile(nullptr), saveFile(nullptr), hostBinData(nullptr), hostSymbolData(nullptr), range(nullptr)
{
    // if (UNCOMPRESSED_PACKET_SIZE % sizeof(READ_ELEMENT) > 0)
    // {
    //     throw string("The input packet size must be the multiple of READ_ELEMENT!");
    // }

    // if (UNCOMPRESSED_PACKET_SIZE >= MAX_PROBABILITY - UPPER(EOF_CHAR))
    // {
    //     throw string("The packet's size was too large that to occur overflow problem");
    // }

    sdkCreateTimer(&this->process_timer);
    sdkCreateTimer(&this->io_timer);
    cudaMallocHost((void **)&this->range, sizeof(AdaptiveProbabilityRange));
}


gip::Compressor::~Compressor()
{
    sdkDeleteTimer(&this->process_timer);
    sdkDeleteTimer(&this->io_timer);
    checkCudaErrors(cudaFreeHost(this->range)); //for both GPU and CPU
}