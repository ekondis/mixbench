CUDA_INSTALL_PATH = /usr/local/cuda
CC = g++
OPTFLAG = -O2
NVCC = ${CUDA_INSTALL_PATH}/bin/nvcc
FLAGS = ${OPTFLAG} -I${CUDA_INSTALL_PATH}/include -Wall
NVFLAGS = -O2 -I${CUDA_INSTALL_PATH}/include --compiler-options -fno-strict-aliasing --ptxas-options=-v -Xptxas -dlcm=cg
BITS = $(shell getconf LONG_BIT)
ifeq (${BITS},64)
	LIBSUFFIX := 64
endif
LFLAGS = ${OMPFLAG} ${PROFLAG} -L${CUDA_INSTALL_PATH}/lib${LIBSUFFIX} -lm -lstdc++ -lcudart -lrt
NVCODE = -gencode=arch=compute_20,code=\"compute_20\"
#NVCODE = -gencode=arch=compute_52,code=\"compute_52\" -gencode=arch=compute_30,code=\"compute_30\" -gencode=arch=compute_20,code=\"compute_20\"
#NVCODE = -gencode=arch=compute_30,code=\"compute_30\"

.PHONY: all

all: mixbench-cuda mixbench-cuda-bs

mixbench-cuda: main.o mix_kernels.o
	${CC} -o $@ $^ ${LFLAGS}

mixbench-cuda-bs: main.o mixbench-cuda-bs.o
	${CC} -o $@ $^ ${LFLAGS}

main.o: main.cpp mix_kernels.h timestamp.h lcutil.h
	${CC} -c ${FLAGS} $<

mix_kernels.o: mix_kernels.cu timestamp.h lcutil.h
	${NVCC} ${NVCODE} ${NVFLAGS} -DUNIX -c $< -o $@

mixbench-cuda-bs.o: mix_kernels.cu
	${NVCC} ${NVCODE} ${NVFLAGS} -DUNIX -DBLOCK_STRIDED -c $< -o $@

clean:
	\rm -f mixbench-cuda main.o mix_kernels.o mixbench-cuda-bs mixbench-cuda-bs.o

rebuild: clean mixbench-cuda mixbench-cuda-bs

