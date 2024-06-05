# GPU_AC: Fast GPU Based Arithmetic Coding

## About

> This is a modified version of the [GPUAR](https://github.com/jiahansu/GPUAR). GPU_AC takes user-defined symbols and their probability distribution from Python as input and can perform parallel and fast entropy coding on the GPU. The backend is written in C++, the API is basic types in ``cytpes``. The implementation is based on [torchAc](https://github.com/fab-jul/torchac), meaning that we implement _arithmetic coding_. We encode a PACKET of symbols with ``UNCOMPRESSED_PACKET_SIZE=8192`` in parallel and append 4 bytes ``[bin size, number of symbs]``at the header of each paket bin.

### Compile

```bash
sh ./compile.sh
```

### Example

```python
import GPU_AC
import torch 

# Generate random symbols and pdf (both must be on the same device/CPU and in uint16).
dim = 256
symsNum = 8192*4 # compress 8192 sybmols in one thread
pdf = torch.rand(symsNum,dim).short().cuda() + 0.01
pdf = pdf / (torch.sum(pdf,1,keepdims=True)) # N*dim
symgpu = torch.randint(0,dim,(symsNum,1)).short().cuda() # max_value of symbols < dim
pdfgpu = pdf

filebin = 'gpuac.bin'
# Encode to bytestream.
encodsz = GPU_AC.encode(symgpu, pdf,filebin,useGpu=True,interaction=True)

# Number of bits taken by the stream.
print('real_bpp',encodsz/symsNum)

# Theoretical bits number
criterion = torch.nn.NLLLoss()
print('shannon entropy', criterion(torch.log2(pdfgpu), symgpu.reshape(-1).long()))

# Decode from bytestream.
symbols_dec = GPU_AC.decode(pdf,filebin,useGpu=True,interaction=True)
assert (symbols_dec == symbols_dec).all()
```

### Performance


|     |         | rate/entropy |           |         |       |         | time(s)   |           |         |       |             |
| --- | ------- | ------------ | --------- | ------- | ----- | ------- | --------- | --------- | ------- | ----- | ----------- |
| dim | symNum  | arcodeGpu    | arcodeCpu | torchAc | yaecl | entropy | arcodeGpu | arcodeCpu | torchAc | yaecl | torchAcw/io |
| 128 | 1000000 | 1.003        | 1.003     | 1.001   | 1.001 | 2.369   | 0.183     | 1.004     | 1.207   | 1.059 | 0.933       |
| 256 | 1000000 | 1.004        | 1.004     | 1.002   | 1.002 | 2.382   | 0.236     | 1.88      | 2.2     | 1.977 | 1.675       |
| 512 | 1000000 | 1.007        | 1.007     | 1.005   | 1.005 | 2.388   | 0.376     | 3.43      | 4.14    | 3.577 | 2.975       |

## Citation

Reference from [GPUAR](https://github.com/jiahansu/GPUAR), [torchac](https://github.com/fab-jul/torchac), thanks!
