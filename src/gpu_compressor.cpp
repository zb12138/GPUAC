#include "gpu_compressor.h"
#include "file_header.h"
#include <cassert>
using namespace gip;

gip::GPUCompressor::GPUCompressor(probability_t *data, int &deviceID, size_t endOfChar) : numBlocks(0), totalThreads(0), uncompressedDataBufferSize(0), inputStreams{}, outputStream(0),
                                                                        devieSymbolData(nullptr), deviceBinData(nullptr), inputPacketBuffer(nullptr), outputPacketBuffer(nullptr)
{
    if (deviceID < 0)
    {
        deviceID = gpuGetMaxGflopsDeviceId();
    }
    /*
    for (int i = 0; i < GPUCompressor::NUM_STREAMS; ++i)
    {
        this->inputStreams[i] = 0;
    }*/
    // assert(0);
    en_of_char = endOfChar;
    this->chooseDevice(deviceID);
    this->range->ranges = data;
    initConstantRange(this->range,endOfChar);
    this->allocateResource();
}

void gip::GPUCompressor::cleanResource()
{

    for (int i = 0; i < GPUCompressor::NUM_STREAMS; ++i)
    {
        if (this->inputStreams[i] > 0)
        {
            checkCudaErrors(cudaStreamDestroy(this->inputStreams[i]));
        }
    }

    if (this->outputStream > 0)
    {
        checkCudaErrors(cudaStreamDestroy(this->outputStream));
    }

    checkCudaErrors(cudaFree(deviceBinData));

    checkCudaErrors(cudaFreeHost(inputPacketBuffer));
    checkCudaErrors(cudaFreeHost(outputPacketBuffer));
    checkCudaErrors(cudaFreeHost(eachPacketBinSize));
}

gip::GPUCompressor::~GPUCompressor()
{

    this->cleanResource();
}

void gip::GPUCompressor::allocateResource()
{

    for (int i = 0; i < GPUCompressor::NUM_STREAMS; ++i)
    {
        checkCudaErrors(cudaStreamCreate(&(this->inputStreams[i])));
    }

    checkCudaErrors(cudaStreamCreate(&(this->outputStream)));

    checkCudaErrors(cudaMalloc((void **)&deviceBinData, COMPRESSED_PACKET_SIZE * this->totalThreads));
    checkCudaErrors(cudaMallocHost((void **)&inputPacketBuffer, COMPRESSED_PACKET_SIZE * NUM_STREAMS));
    checkCudaErrors(cudaMallocHost((void **)&outputPacketBuffer, COMPRESSED_PACKET_SIZE));
    checkCudaErrors(cudaMallocHost((void **)&eachPacketBinSize, this->totalThreads));
}

void gip::GPUCompressor::chooseDevice(int deviceID)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceID);
    cudaChooseDevice(&deviceID, &prop);
    checkCudaErrors(cudaSetDevice(deviceID));
    this->numBlocks = prop.multiProcessorCount * PROCESSORS_GRIDS_FACTOR;
    this->totalThreads = NUM_THREADS * this->numBlocks;
    this->uncompressedDataBufferSize = UNCOMPRESSED_PACKET_SIZE * this->totalThreads;
}

CompressionInfo gip::GPUCompressor::compress(ProgressMonitor *monitor, symb_t *symbols, size_t symbol_Num)
{
    CompressionInfo info;
    saveFile = fopen(this->saveFileName.c_str(), "wb");
    size_t size = 0;
    FileHeader fileHeader;
    cudaError_t result;
    uint32_t inputStreamIndex;
    uint32_t inputBufferOffset;
    uint32_t numPackets = 0;
    uint32_t packetSize;

    uint32_t writeOutNumPackets;
    uint32_t blocks;
    uint32_t outNumPackets;
    uint32_t packet_id = 0;
    uint32_t symbol_id = 0;
    uint32_t symbol_procs_num_max = UNCOMPRESSED_PACKET_SIZE * this->totalThreads;
    try
    {
        monitor->reset();
        info.uncompressedFileSize = symbol_Num * ceil(log2(en_of_char));
        info.compressedFileSize = FileHeader::HEADER_LENGTH;

        assert(saveFile != NULL && "Open output file failed.");

        fseek(saveFile, FileHeader::HEADER_LENGTH, SEEK_SET); // reserve for header
        // printf("\n\nencoding...symbol_Num %d \n",symbol_Num);
        while (symbol_id < symbol_Num)
        {
            uint32_t symbol_process_num = MIN(symbol_Num - symbol_id, symbol_procs_num_max);
            devieSymbolData = symbols + symbol_id;
            outNumPackets = ceil((double)symbol_process_num / ((double)UNCOMPRESSED_PACKET_SIZE));
            blocks = ceil((double)outNumPackets / ((double)NUM_THREADS));
            // printf("en:outNumPackets %d , blocks %d symbol_procs_num_max  %d\n ", outNumPackets, blocks, symbol_procs_num_max);

            cutilCheckError(sdkStartTimer(&process_timer));
            garCompressExecutor((const symb_t *)devieSymbolData, symbol_process_num, deviceBinData, blocks, eachPacketBinSize); // get into gpu compressor
            result = cudaDeviceSynchronize();
            if (result != cudaSuccess)
            {
                throw string("Fail to execute kernel code: ") + string(cudaGetErrorString(result));
            }
            // printf("compress ok, calc bits...\n");
            cutilCheckError(sdkStopTimer(&process_timer));

            cutilCheckError(sdkStartTimer(&io_timer));

            for (size_t i = 0; i < outNumPackets; i++)
            {
                packetSize = eachPacketBinSize[i];
                // printf("outNumPacketsID %ld, packetSize %d\n",i,packetSize);
                cudaMemcpyAsync(outputPacketBuffer, deviceBinData + i * COMPRESSED_PACKET_SIZE, packetSize, cudaMemcpyDeviceToHost, this->outputStream);

                cudaStreamSynchronize(this->outputStream);
                assert((fwrite(outputPacketBuffer, packetSize, 1, saveFile) > 0) && "Write compressed data to output file failed");
                info.compressedFileSize += packetSize*8;
            }
            cutilCheckError(sdkStopTimer(&io_timer));

            symbol_id += symbol_process_num;
        }
        cutilCheckError(sdkStartTimer(&io_timer));

        // write header
        fileHeader.setCompressedFileSize(info.compressedFileSize);
        fileHeader.setUncompressedFileSize(info.uncompressedFileSize);

        fseek(saveFile, 0, SEEK_SET);
        fwrite(fileHeader.getData(), FileHeader::HEADER_LENGTH, 1, saveFile);
        cutilCheckError(sdkStopTimer(&io_timer));
    }
    catch (exception e)
    {
        this->closeFiles();
        throw;
    }
    cutilCheckError(sdkStartTimer(&io_timer));
    this->closeFiles();
    cutilCheckError(sdkStopTimer(&io_timer));

    info.processTime = sdkGetTimerValue(&process_timer);
    info.ioTime = sdkGetTimerValue(&io_timer);
    return info;
}

CompressionInfo gip::GPUCompressor::decompress(ProgressMonitor *monitor, symb_t *symbols_dec, size_t symbol_Num)
{
    CompressionInfo info;
    openFile = fopen(this->openFileName.c_str(), "rb");
    size_t readSize = 0;
    size_t size = 0;
    FileHeader fileHeader;
    cudaError_t result;
    uint32_t inputStreamIndex;
    uint32_t inputBufferOffset = 0;
    uint32_t numPackets = 0;
    uint32_t packetlength = 0;
    uint32_t outNumPackets;
    uint32_t writeOutNumPackets;
    uint32_t blocks;
    uint32_t maxCompressedPackets = NUM_THREADS * this->numBlocks;
    size_t r;
    uint32_t packet_id = 0;
    uint32_t symbol_id = 0;
    uint32_t symbol_procs_num_max = UNCOMPRESSED_PACKET_SIZE * this->totalThreads;

    monitor->reset();
    assert(openFile != NULL && "Open input file failed.");
    fseek(openFile, 0, SEEK_SET);
    assert(fread(fileHeader.getData(), FileHeader::HEADER_LENGTH, 1, openFile) > 0 && "Incorrect file format");
    assert(fileHeader.checkHeaderVersion() && "Incorrect file format");
    // readSize = FileHeader::HEADER_LENGTH;
    info = fileHeader.getInfo();

    // std::swap(devieSymbolData, deviceBinData);
    // printf("\n\n\n\ndecoding... ");
    while (symbol_id < symbol_Num)
    {
        uint32_t symbol_process_num = MIN(symbol_Num - symbol_id, symbol_procs_num_max);
        inputStreamIndex = 0;
        devieSymbolData = symbols_dec + symbol_id;
        outNumPackets = ceil((double)symbol_process_num / ((double)UNCOMPRESSED_PACKET_SIZE));
        blocks = ceil((double)outNumPackets / ((double)NUM_THREADS));
        // printf("209: outNumPackets %d , blocks %d symbol_Num %ld\n", outNumPackets, blocks, symbol_Num);

        cutilCheckError(sdkStartTimer(&io_timer));
        numPackets = 0;
        while (outNumPackets > numPackets)
        {
            inputBufferOffset = inputStreamIndex * COMPRESSED_PACKET_SIZE;
            // inputBufferOffset = packetlength + inputBufferOffset + inputStreamIndex * COMPRESSED_PACKET_SIZE;
            fread(inputPacketBuffer + inputBufferOffset, PACKET_HEADER_LENGTH, 1, openFile);
            packetlength = getCompressedSize(inputPacketBuffer + inputBufferOffset); // read(compressed,2);
            // printf("packetlength %d \n",packetlength);
            r = fread(inputPacketBuffer + inputBufferOffset + PACKET_HEADER_LENGTH, sizeof(uint8_t), packetlength - PACKET_HEADER_LENGTH, openFile);
            // printf("packetlength %d r %d numPackets%d outNumPackets%d\n",packetlength,r,numPackets,outNumPackets);
            // assert((r == packetlength - PACKET_HEADER_LENGTH) && "Invalid file length");
            char *unz = inputPacketBuffer + inputBufferOffset + PACKET_HEADER_LENGTH;
            // printf("%x,%x,%x",unz[0],unz[1],unz[2]);
            // printf("id :%d\n",(numPackets * COMPRESSED_PACKET_SIZE));
            checkCudaErrors(cudaMemcpy(deviceBinData + (numPackets * COMPRESSED_PACKET_SIZE), inputPacketBuffer + inputBufferOffset, packetlength, cudaMemcpyHostToDevice));
            ++numPackets;
            // readSize += packetlength;
            // inputStreamIndex = (inputStreamIndex + 1) % NUM_STREAMS;
        }
        cudaStreamSynchronize(this->inputStreams[0]);
        // for (int i = 0; i < GPUCompressor::NUM_STREAMS; ++i)
        // {
        //     cudaStreamSynchronize(this->inputStreams[i]);
        // }
        cutilCheckError(sdkStopTimer(&io_timer));

        cutilCheckError(sdkStartTimer(&process_timer));
        if (outNumPackets > 0)
        {
            garDecompressExecutor(deviceBinData, symbol_process_num, devieSymbolData, blocks); // get into gpu compressor
            result = cudaDeviceSynchronize();
            if (result != cudaSuccess)
            {
                throw string("Fail to execute kernel code: ") + string(cudaGetErrorString(result));
            }

            // printdata<uint8_t>(10,devieOutputData,true,0);
        }
        cutilCheckError(sdkStopTimer(&process_timer));
        symbol_id += symbol_process_num;
    }
    cutilCheckError(sdkStartTimer(&io_timer));
    this->closeFiles();
    cutilCheckError(sdkStopTimer(&io_timer));

    info.processTime = sdkGetTimerValue(&process_timer);
    info.ioTime = sdkGetTimerValue(&io_timer);
    // std::swap(devieSymbolData, deviceBinData);
    return info;
}