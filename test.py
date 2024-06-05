import GPU_AC
import torch 

# Generate random symbols and pdf (both must be on the same device/CPU and in uint16).
dim = 256
symsNum = 8192*4 # compress 8192 sybmols in one thread
pdf = torch.rand(symsNum,dim).short().cuda() + 0.01
pdf = pdf / (torch.sum(pdf,1,keepdims=True))
symgpu = torch.randint(0,dim,(symsNum,1)).short().cuda()
pdfgpu = pdf

filebin = 'gpuac.bin'
# Encode to bytestream.
encodsz = GPU_AC.encode(symgpu, pdf,filebin,useGpu=False,interaction=True)

# Number of bits taken by the stream.
print('real_bpp',encodsz/symsNum)

# Theoretical bits number
criterion = torch.nn.NLLLoss()
print('shannon entropy', criterion(torch.log2(pdfgpu), symgpu.reshape(-1).long()))

# Decode from bytestream.
symbols_dec = GPU_AC.decode(pdf,filebin,useGpu=False,interaction=True)
assert (symbols_dec == symbols_dec).all()