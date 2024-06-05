if [ ! -d "build" ]; then
  mkdir build
fi
cd build
c++ -I../src -I/usr/local/cuda/include  -fPIC -std=c++14 -c ../src/progress_monitor.cpp -o progress_monitor.o  
c++ -I../src -I/usr/local/cuda/include  -fPIC -std=c++14 -c ../src/cpu_compressor.cpp -o cpu_compressor.o  
c++ -I../src -I/usr/local/cuda/include  -fPIC -std=c++14 -c ../src/gpuac_backend.cpp -o gpuac_backend.o  
c++ -I../src -I/usr/local/cuda/include  -fPIC -std=c++14 -c ../src/compressor.cpp -o compressor.o 
c++ -I../src -I/usr/local/cuda/include  -fPIC -std=c++14 -c ../src/gpu_compressor.cpp -o gpu_compressor.o  
nvcc -I../src -I/usr/local/cuda/include --expt-relaxed-constexpr -gencode=arch=compute_61,code=compute_61 -gencode=arch=compute_61,code=sm_61 --compiler-options '-fPIC' -std=c++14 -c ../src/gpuar_kernel.cu -o gpuar_kernel.cuda.o -O4 --use_fast_math 
c++ gpuac_backend.o gpuar_kernel.cuda.o gpu_compressor.o cpu_compressor.o compressor.o progress_monitor.o -shared -L/usr/local/cuda/lib64 -lcudart -o gpuac_backend.so -Ofast
cd ..