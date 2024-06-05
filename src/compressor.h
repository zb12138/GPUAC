#pragma once

#include "compress_info.h"
#include "progress_monitor.h"

using namespace std;

namespace gip
{
    class Compressor
    {
    protected:
        string openFileName;
        string saveFileName;
        StopWatchInterface *process_timer;
        StopWatchInterface *io_timer;
        FILE *openFile;
        FILE *saveFile;
        unsigned char *hostBinData;
        symb_t *hostSymbolData;
        AdaptiveProbabilityRange *range;
        size_t en_of_char;
        size_t cdf_dim;
    public:
        virtual CompressionInfo compress(ProgressMonitor *monitor, symb_t *symbs, size_t symbs_num) = 0;
        virtual CompressionInfo decompress(ProgressMonitor *monitor, symb_t *symbs, size_t symbs_num) = 0;
        Compressor();
        virtual ~Compressor();

        size_t getFileSize(FILE *stream)
        {
            size_t t;
            fseek(openFile, 0, SEEK_SET);
            t = ftell(openFile);
            fseek(openFile, 0, SEEK_END);
            t = ftell(openFile) - t;

            return t;
        }
        void setOpenFileName(const string fileName)
        {
            this->openFileName = fileName;
        }
        void setSaveFileName(const string fileName)
        {
            this->saveFileName = fileName;
        }
        void closeFiles()
        {
            if (this->saveFile != NULL)
            {
                fclose(this->saveFile);
            }
            if (this->openFile != NULL)
            {
                fclose(this->openFile);
            }
        }
    };
}