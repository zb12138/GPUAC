
// #include <torch/extension.h>
#include "cpu_compressor.h"
#include "gpu_compressor.h"

using namespace gip;
using namespace std;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

// template <typename T, typename T2>
// T *getTensorData(torch::Tensor &tensor, size_t &s1, size_t &s2)
// {
//     const auto s = tensor.sizes();
//     TORCH_CHECK(s.size() == 2, "Invalid size for cdf! Expected (N, Lp)")
//     const auto cdf_acc = tensor.accessor<T2, 2>();
//     s1 = s[0];
//     s2 = s[1];
//     return (T *)cdf_acc.data();
// }

int ac_codec(const char *inputName, const char *outputName, symb_t *symbs, probability_t *cdf,size_t symbols_num, size_t cdf_dim, bool isCompress, bool hostMode, bool interactive)
{
    try
    {
        bool showeHelp = false;
        int deviceID = -1;
        bool hasInput = true;
        bool hasOutput = true;
        int numDevices;
        string deviceName;
        {
            // if (cdf.is_cuda() == hostMode)
            // {
            //     if (hostMode)
            //         printf("Warnning, found cdf on GPU, run on GPU.\n");
            //     else
            //         printf("Warnning, found cdf on CPU, run on CPU.\n");
            //     hostMode = !hostMode;
            // }
            if (!hostMode)
            {
                cudaGetDeviceCount(&numDevices);
                if (interactive)
                    printf("Fund %d devices. \n", numDevices);
                if (numDevices == 0)
                {
                    printf("Warnning, no GPU found, run on CPU. \n");
                    hostMode = true;
                }
            }
            ProgressMonitor *monitor = new ProgressMonitor();
            Compressor *compressor = NULL;
            CompressionInfo info;
            
            // probability_t *prob;
            // symb_t *symbs;
            // size_t symbols_num;
            // symbs = getTensorData<symb_t, int16_t>(symb, symbols_num, prob_dims);
            // prob = getTensorData<probability_t, int16_t>(cdf, symbols_num, cdf_dim);
            size_t end_of_char = cdf_dim - 2;
            
            // assert(prob_dims == CDF_DIM);
            if (hostMode || numDevices <= 0)
            {
                compressor = new CPUCompressor(cdf,deviceID,end_of_char);
                deviceName = "CPU";
            }
            else
            {
                // TORCH_CHECK(cdf.is_cuda(), "cdf must be on GPU!")
                compressor = new GPUCompressor(cdf, deviceID,end_of_char);
                deviceName = "GPU: " + std::to_string(deviceID);
            }
            
            if (isCompress)
            {
                // printf("%s",outputName);
                compressor->setSaveFileName(outputName);
                if (interactive)
                    cout << "Start to compress " << inputName << " to " << outputName << " on " << deviceName << "." << endl;
                info = compressor->compress(monitor, symbs, symbols_num);
            }
            else
            {
                compressor->setOpenFileName(inputName);
                if (interactive)
                    cout << "Start to decompress " << inputName << " to " << outputName << " on " << deviceName << "." << endl;
                info = compressor->decompress(monitor, symbs, symbols_num);
            }

            if (interactive)
            {
                double compressionRatio = (double)info.compressedFileSize / info.uncompressedFileSize;
                cout << "Complete" << endl;
                cout << "Statistics: " << endl;
                cout << "Uncompressed file size " << info.uncompressedFileSize << " bits" << endl;
                cout << "Compressed file size  " << info.compressedFileSize << " bits" << endl;
                cout << "Compression ratio     " << compressionRatio << endl;
                cout << "Compute time          " << info.processTime / 1000 << " s" << endl;
                cout << "I/O time              " << info.ioTime / 1000 << " s" << endl;
                cout << "Score                 " << (1000 / (pow(compressionRatio, 0.6) * pow(info.processTime / 1000, 0.4))) << endl
                     << endl;
                // compressor->generateRandomFile(64*1024*1024);
            }
            delete monitor;
            delete compressor;
            compressor = NULL;
            monitor = NULL;
            return info.compressedFileSize;
        }
    }
    catch (char *e)
    {
        cerr << e << endl;
    }
    catch (string s)
    {
        cerr << s << endl;
    }
}
extern "C" {
    int encode(const char *compressedFile, symb_t *symb, probability_t *cdf, size_t symbols_num, size_t cdf_dim, bool onDevice, bool interactive)
    {
        return ac_codec("uncompressedFile", compressedFile, symb, cdf, symbols_num, cdf_dim, true, !onDevice, interactive);
    }

    int decode(symb_t * symb, const char *compressedFile, probability_t *cdf, size_t symbols_num, size_t cdf_dim,bool onDevice, bool interactive)
    {
        return ac_codec(compressedFile, "uncompressedFile", symb, cdf, symbols_num, cdf_dim, false, !onDevice, interactive);
    }
}
// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
// {
//     m.def("encode", &encode, "encode");
//     m.def("decode", &decode, "decode");
// }
