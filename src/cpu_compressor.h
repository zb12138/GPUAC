#pragma once

#include "compressor.h"

using namespace std;

namespace gip
{
    class CPUCompressor : public Compressor
    {
    public:
        CPUCompressor(probability_t* data, int& deviceID, size_t endOfChar);
        ~CPUCompressor();
        CompressionInfo compress(ProgressMonitor *monitor,  symb_t *symbs, size_t symbs_num);
        CompressionInfo decompress(ProgressMonitor *monitor, symb_t *symbs, size_t symbs_num);
    };
}