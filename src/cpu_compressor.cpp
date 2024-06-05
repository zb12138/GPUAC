#include "cpu_compressor.h"
#include "file_header.h"

using namespace gip;

gip::CPUCompressor::CPUCompressor(probability_t *data, int &deviceID, size_t endOfChar)
{
    en_of_char = endOfChar;
    cdf_dim = endOfChar+2;
    this->range->ranges = data;
    initConstantRange(this->range, en_of_char);
    cudaMallocHost((void **)&hostBinData, COMPRESSED_PACKET_SIZE);
}


gip::CPUCompressor::~CPUCompressor()
{
    checkCudaErrors(cudaFreeHost(hostBinData));
}

CompressionInfo gip::CPUCompressor::compress(ProgressMonitor *monitor, symb_t *symbols, size_t symbol_Num)
{
    
    CompressionInfo info;
    saveFile = fopen(this->saveFileName.c_str(), "wb");
    FileHeader fileHeader;
    size_t r;
    size_t symbol_id = 0;
    probability_t *cdf = range->ranges;
    try
    {
        monitor->reset();
        if(!(saveFile != NULL)) throw std::runtime_error("x is not 10");
        fseek(saveFile, FileHeader::HEADER_LENGTH, SEEK_SET); // reserve for header
        info.compressedFileSize = FileHeader::HEADER_LENGTH;
        info.uncompressedFileSize = symbol_Num * ceil(log2(en_of_char));
        
        while (symbol_id < symbol_Num)
        {
            cutilCheckError(sdkStartTimer(&process_timer));
            uint32_t symbol_process_num = MIN(symbol_Num - symbol_id, UNCOMPRESSED_PACKET_SIZE);
            const symb_t * hostSymbolData2 = symbols + symbol_id;
            cdf = range->ranges + symbol_id*cdf_dim;            
            r = arCompress((const symb_t *)hostSymbolData2, symbol_process_num, (uint8_t *)hostBinData, cdf);
            cutilCheckError(sdkStopTimer(&process_timer));
            
            cutilCheckError(sdkStartTimer(&io_timer));
            assert(fwrite(hostBinData, r, 1, saveFile) >0 && "Write data to file failed");
            symbol_id += symbol_process_num;
            cutilCheckError(sdkStopTimer(&io_timer));
            info.compressedFileSize += r*8;
            // monitor->updateProgress(&info);
        }
        cutilCheckError(sdkStartTimer(&io_timer));
        fseek(saveFile, 0, SEEK_SET);
        fileHeader.setCompressedFileSize(info.compressedFileSize);
        fileHeader.setUncompressedFileSize(info.uncompressedFileSize);
        assert (fwrite(fileHeader.getData(), FileHeader::HEADER_LENGTH, 1, saveFile) > 0 && "Write header failed.");
        cutilCheckError(sdkStopTimer(&io_timer));
    }
    catch (exception e)
    {
        std::cout << e.what() << std::endl;
        this->closeFiles();
        throw;
    }
    this->closeFiles();
    info.processTime = sdkGetTimerValue(&process_timer);
    info.ioTime = sdkGetTimerValue(&io_timer);
    return info;
}


CompressionInfo gip::CPUCompressor::decompress(ProgressMonitor *monitor, symb_t *symbols_dec, size_t symbol_Num)
{
    CompressionInfo info;

    openFile = fopen(this->openFileName.c_str(), "rb");
    saveFile = fopen(this->saveFileName.c_str(), "wb");
    int size = 0;
    FileHeader fileHeader;
    size_t r;
    unsigned int packetSize;
    probability_t cumProb;
    uint32_t symbol_id = 0;
    probability_t *cdf = range->ranges;

    monitor->reset();
    try
    {   
        cutilCheckError(sdkStartTimer(&io_timer));
        assert(openFile != NULL && "Open input file failed.");
        fseek(openFile, 0, SEEK_SET);
        assert(fread(fileHeader.getData(), FileHeader::HEADER_LENGTH, 1, openFile) > 0 && "Incorrect file format");
        assert(fileHeader.checkHeaderVersion() && "Incorrect file format");
        info = fileHeader.getInfo();
        cutilCheckError(sdkStopTimer(&io_timer));
        
        while (symbol_id < symbol_Num)
        {

            cutilCheckError(sdkStartTimer(&io_timer));
            uint32_t symbol_process_num = MIN(symbol_Num - symbol_id, UNCOMPRESSED_PACKET_SIZE);
            hostSymbolData = symbols_dec + symbol_id;
            cdf = range->ranges + symbol_id*cdf_dim;
            fread(hostBinData, PACKET_HEADER_LENGTH, 1, openFile);
            packetSize = getCompressedSize(hostBinData); // read(compressed,2);
            fread(hostBinData + PACKET_HEADER_LENGTH, packetSize - PACKET_HEADER_LENGTH, 1, openFile);
            cutilCheckError(sdkStopTimer(&io_timer));

            cutilCheckError(sdkStartTimer(&process_timer));
            BitPointer bfpIn = createBitPointer((uint8_t *)hostBinData + PACKET_HEADER_LENGTH);
            bfpIn.in_sz = getCompressedSize(hostBinData);
            // uint16_t r = getUncompressedSize(hostBinData);
            // uint16_t r = 
            arDecompress(bfpIn,symbol_process_num, (unsigned symb_t *)hostSymbolData,cdf);
            symbol_id += symbol_process_num;
            cutilCheckError(sdkStopTimer(&process_timer));
            // info.processedUncompressedSize += r; 
            // monitor->updateProgress(&info);
        }
    }
    catch (exception e)
    {
        this->closeFiles();
        throw;
    }
    this->closeFiles();
    info.processTime = sdkGetTimerValue(&process_timer);
    info.ioTime = sdkGetTimerValue(&io_timer);
    return info;
}
