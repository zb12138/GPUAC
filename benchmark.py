import torch 
import torch.nn as nn
import torch.nn.functional as F
import time 
import os 
import GPU_AC
from GPU_AC import write_bin,read_bin,BuildProbabilityRangeList
import numpy as np 
import time
import torchac
import pandas as pd
import yaecl
import torch
import numpy as np
import re 

# dim = 512 
def yaeclActest(sym, pdf):
    filebin = 'yaecl.bin'
    sym = sym.numpy().astype(np.int32)
    cdf = BuildProbabilityRangeList(pdf).numpy().astype(np.uint16).astype(np.uint32) 
    cdf[:,-1] = 2**16
    cdf[:,0] = 0
    ac_enc = yaecl.ac_encoder_t()
    symd_b = np.zeros_like(sym, dtype=np.int32)
    ac_enc.encode_nxn(sym, memoryview(cdf), 16)
    ac_enc.flush()
    ac_enc.bit_stream.save(filebin)
    bs = yaecl.bit_stream_t()
    bs.load(filebin)
    ac_dec = yaecl.ac_decoder_t(bs)
    ac_dec.decode_nxn(pdf.shape[1],  memoryview(cdf), 16, symd_b)
    # assert (symd_b==sym).all()
    return os.stat(filebin).st_size*8

def torchActest(sym, pdf):
    filebin = 'artorah.bin'
    sym = sym.reshape(-1)
    cdfF = torch.cumsum(pdf, axis=1)
    cdfF = torch.hstack((torch.zeros((pdf.shape[0], 1)), cdfF))
    cdfF = torch.clamp(cdfF/cdfF[:, -1:], 0, 1)
    code = torchac.encode_float_cdf(cdfF, sym.short(), check_input_bounds=True)
    write_bin(code, filebin)
    decode_sym = torchac.decode_float_cdf(cdfF, bytes(read_bin(filebin)))
    # assert (decode_sym == sym).all()
    return os.stat(filebin).st_size*8

def arcodeGputest(symbols, pdf):
    filebin = 'arGpu.bin'
    encodsz = GPU_AC.encode(symbols, pdf,filebin,interaction=False)
    # assert (encodsz==os.stat(filebin).st_size*8)
    symbols_dec = GPU_AC.decode(pdf,filebin)
    # assert (symbols_dec == symbols).all()
    return os.stat(filebin).st_size*8,symbols_dec


def arcodeCputest(symbols, pdf):
    filebin = 'arCpu.bin'
    encodsz = GPU_AC.encode(symbols, pdf,filebin,useGpu=False)
    # assert (encodsz==os.stat(filebin).st_size*8)
    symbols_dec = GPU_AC.decode(pdf,filebin,useGpu=False)
    # 
    return os.stat(filebin).st_size*8 

def randNN(dim, symsNum,type):
    syms = np.random.randint(0, dim,(symsNum,1)).astype(np.int16)
    if type in ['1','2']:
        if type=='1':
            mu2 = syms
            sigma2 = 5#dim/6
        if type == '2':
            mu2 = dim/2
            sigma2 = dim/6
        pdf = 1/((np.pi*2)**0.5*sigma2) * np.exp(  (-(np.tile(np.arange(dim), (symsNum, 1))-mu2)**2/(2*sigma2**2))) 
    if type == '3':
        pdf = np.ones((symsNum,dim))
    pdf = np.clip(pdf/(np.sum(pdf, 1, keepdims=True)),0,1)
    return syms,pdf 

def printFun(value,print=print):
    if print is not None:
        s= str(value)
        print(re.sub(r'(\d+)(.\d{1,3})(\d*)', r'\1\2',s))

criterion = torch.nn.NLLLoss()
if __name__ == '__main__':
    for dim in [128,256,512,1024]:
        results = []
        for pdf_type in ['1','2','3']:
            symsNum = 1000000
            for i in range(2):
                syms, pdf = randNN(dim,symsNum,pdf_type)
                pdfcpu = torch.Tensor(pdf)  
                symcpu = torch.Tensor(syms)
                pdfgpu = pdfcpu.cuda()
                symgpu = symcpu.cuda()
            
                shannon_entropy, gpuAc_bits, torchAc_bits, cpuAc_bits,yaecl_bits = 0, 0, 0, 0,0
                gpuAc_t, torchAc_t, cpuAc_t,yaecl_t,gpuAc_woIO_t = 0, 0, 0,0, 0
                shannon_entropy += criterion(torch.log(pdfgpu), symgpu.reshape(-1).long())/np.log(2)*symsNum

                t = time.time()
                yaecl_bits += yaeclActest(symcpu, pdfcpu)
                yaecl_t += (time.time() - t)

                t = time.time()
                bits,symbols_dec = arcodeGputest(symgpu, pdfgpu)
                gpuAc_bits+=bits 
                gpuAc_t += (time.time() - t)
                
                assert (symbols_dec == symgpu).all()

                t = time.time()
                torchActest(symcpu, pdfcpu)
                gpuAc_woIO_t += (time.time() - t)

                t = time.time()
                torchAc_bits += torchActest(symgpu.cpu(), pdfgpu.cpu())
                torchAc_t += (time.time() - t)

                t = time.time()
                cpuAc_bits += arcodeCputest(symcpu, pdfcpu)
                cpuAc_t+=(time.time() - t)

                symsT = 1#symsNum/1024/1024
                entropy = shannon_entropy.cpu().numpy()
                result = {'dim':dim,'pdf':pdf_type,'sym':symsNum, 'gpuAc_ef': gpuAc_bits/entropy, 'cpuAc_ef':cpuAc_bits/entropy,'torchAc_ef': torchAc_bits/entropy, 'yaecl_ef':yaecl_bits/entropy, \
                            "entropy": entropy/symsNum, "arcodeGpu": gpuAc_t/symsT,"arcodeCpu": cpuAc_t/symsT, "torchAc":  torchAc_t/symsT, "torchAcw/io":gpuAc_woIO_t, "yaecl": yaecl_t/symsT}
                
                printFun(result)
                results.append(result)

        table = pd.DataFrame(results).round(4)
        table = table.append(table.mean(numeric_only=True),ignore_index=True)
        table.iloc[-1,0] = 'ave'#
        table = table.round(3)
        table.to_csv('entropy_coder{}.csv'.format(dim))
        print(table)