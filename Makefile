CUDA_INSTALL_PATH = /usr/local/cuda
CC = g++
OPTFLAG = -O2
NVCC = ${CUDA_INSTALL_PATH}/bin/nvcc
FLAGS_CUDA = ${OPTFLAG} -I${CUDA_INSTALL_PATH}/include -Wall
NVFLAGS = -O2 -I${CUDA_INSTALL_PATH}/include --compiler-options -fno-strict-aliasing --ptxas-options=-v -Xptxas -dlcm=cg
BITS = $(shell getconf LONG_BIT)
ifeq (${BITS},64)
	LIBSUFFIX := 64
endif
LFLAGS_CUDA = ${OMPFLAG} ${PROFLAG} -L${CUDA_INSTALL_PATH}/lib${LIBSUFFIX} -lm -lstdc++ -lcudart -lrt
NVCODE = -gencode=arch=compute_20,code=\"compute_20\"
#NVCODE = -gencode=arch=compute_52,code=\"compute_52\" -gencode=arch=compute_30,code=\"compute_30\" -gencode=arch=compute_20,code=\"compute_20\"
#NVCODE = -gencode=arch=compute_30,code=\"compute_30\"

.PHONY: all

all: mixbench-cuda mixbench-cuda-bs

mixbench-cuda: main-cuda.o mix_kernels_cuda.o
	${CC} -o $@ $^ ${LFLAGS_CUDA}

mixbench-cuda-bs: main-cuda.o mix_kernels_cuda-bs.o
	${CC} -o $@ $^ ${LFLAGS_CUDA}

main-cuda.o: main-cuda.cpp mix_kernels_cuda.h timestamp.h lcutil.h
	${CC} -c ${FLAGS_CUDA} $<

mix_kernels_cuda.o: mix_kernels_cuda.cu timestamp.h lcutil.h
	${NVCC} ${NVCODE} ${NVFLAGS} -DUNIX -c $< -o $@

mix_kernels_cuda-bs.o: mix_kernels_cuda.cu
	${NVCC} ${NVCODE} ${NVFLAGS} -DUNIX -DBLOCK_STRIDED -c $< -o $@

clean:
	\rm -f mixbench-cuda main-cuda.o mix_kernels_cuda.o mix_kernels_cuda-bs.o mixbench-cuda-bs mixbench-cuda-bs.o

rebuild: clean mixbench-cuda mixbench-cuda-bs

