import ctypes
import torch
import subprocess
import numpy as np 

GPU_AC = ctypes.cdll.LoadLibrary('build/gpuac_backend.so')
symb_t = ctypes.c_uint16
probability_t = ctypes.c_uint16
GPU_AC.encode.argtypes = [
    ctypes.c_char_p,  # compressedFile
    ctypes.POINTER(symb_t),  # symb N*1 torch.Tensor
    ctypes.POINTER(probability_t),  # cdf N*(MAX_VLE+2) torch.Tensor
    ctypes.c_size_t,  # symbols_num
    ctypes.c_size_t,  # cdf_dim
    ctypes.c_bool,  # onDevice
    ctypes.c_bool,  # interactive
]
GPU_AC.encode.restype = ctypes.c_uint

PRECISION = 16

def BuildProbabilityRangeList(pdf):
    cdfF = torch.cumsum(pdf, axis=1)
    cdf_float = torch.clamp(cdfF/cdfF[:, -1:], 0, 1)
    end_of_symb = pdf.shape[-1]
    new_max_value = 1 << (PRECISION)
    new_max_value = new_max_value - end_of_symb
    cdf_float = cdf_float.mul(new_max_value).round()
    cdf = cdf_float.to(dtype=torch.int16, non_blocking=True)
    r = torch.arange(end_of_symb+1, dtype=torch.int16, device=cdf.device)
    pro_start = torch.zeros(
        (cdf.shape[0], 1), dtype=torch.int16, device=cdf.device)
    cdfF = torch.hstack((pro_start, cdf))
    cdfF.add_(r)
    return cdfF.short().contiguous() 

def encode(symbols, pdf, binPath, useGpu=True, interaction = False):
    assert len(symbols) == len(pdf)
    filebin = binPath.encode()
    cnt = BuildProbabilityRangeList(pdf)         #.cuda() 
    symbols = symbols.short().reshape(-1, 1).contiguous() 
    if useGpu:
        symbols = symbols.cuda()
        cnt = cnt.cuda()
    else:
        symbols = symbols.cpu()
        cnt = cnt.cpu()
    symb_ptr = ctypes.cast(symbols.data_ptr(), ctypes.POINTER(ctypes.c_uint16))
    cnt_ptr = ctypes.cast(cnt.data_ptr(), ctypes.POINTER(ctypes.c_uint16))    
    bin_sz_bit = GPU_AC.encode(filebin, symb_ptr, cnt_ptr,cnt.shape[0],cnt.shape[1],useGpu, interaction)
    return bin_sz_bit

def decode(pdf, binPath, useGpu=True, interaction = False): 
    filebin = binPath.encode()
    cnt = BuildProbabilityRangeList(pdf) 
    symb_num  = cnt.shape[0]
    symbols_dec = torch.zeros((symb_num,1)).short() 
    if useGpu:
        symbols_dec = symbols_dec.cuda()
        cnt = cnt.cuda()
    else:
        symbols_dec = symbols_dec.cpu()
        cnt = cnt.cpu()
    symb_dec_ptr = ctypes.cast(symbols_dec.data_ptr(), ctypes.POINTER(ctypes.c_uint16))
    cnt_ptr = ctypes.cast(cnt.data_ptr(), ctypes.POINTER(ctypes.c_uint16))    
    GPU_AC.decode(symb_dec_ptr, filebin,cnt_ptr, symb_num,cnt.shape[1],useGpu, interaction)
    return symbols_dec

def calc_md5(filename):
    output = subprocess.check_output('md5sum {}'.format(filename), shell=True)
    return (output.decode().split(' ')[0])


def md5test(infile, outfile):
    assert calc_md5(infile) == calc_md5(outfile)


def write_bin(code, bitstream_path):
    with open(bitstream_path, 'wb') as fout:
        if type(code) is np.ndarray:
            code = code.tolist()
        fout.write(bytes(code))


def read_bin(bitstream_path, bintype=''):
    with open(bitstream_path+bintype, 'rb') as fin:
        bitstream = fin.read()
    return list(bitstream)