CUDA_INSTALL_PATH = /usr/local/cuda
OCL_INSTALL_PATH = /opt/AMDAPPSDK
OCL_INC_PATH = ${OCL_INSTALL_PATH}/include
OCL_LIB_PATH = ${OCL_INSTALL_PATH}/lib/x86_64
CC = g++
OPTFLAG = -O2
NVCC = ${CUDA_INSTALL_PATH}/bin/nvcc
FLAGS_CUDA = ${OPTFLAG} -I${CUDA_INSTALL_PATH}/include -Wall
FLAGS_OCL = ${OPTFLAG} -I${OCL_INC_PATH} -Wall
NVFLAGS = -O2 -I${CUDA_INSTALL_PATH}/include --compiler-options -fno-strict-aliasing --ptxas-options=-v -Xptxas -dlcm=cg
BITS = $(shell getconf LONG_BIT)
ifeq (${BITS},64)
	LIBSUFFIX_CUDA := 64
endif
LFLAGS_CUDA = -L${CUDA_INSTALL_PATH}/lib${LIBSUFFIX_CUDA} -lm -lstdc++ -lcudart -lrt
LFLAGS_OCL = -L${OCL_LIB_PATH} -lm -lstdc++ -lOpenCL -lrt
NVCODE = -gencode=arch=compute_20,code=\"compute_20\"
#NVCODE = -gencode=arch=compute_52,code=\"compute_52\" -gencode=arch=compute_30,code=\"compute_30\" -gencode=arch=compute_20,code=\"compute_20\"
#NVCODE = -gencode=arch=compute_30,code=\"compute_30\"

.PHONY: all

all: mixbench-cuda mixbench-cuda-bs mixbench-ocl

mixbench-cuda: main-cuda.o mix_kernels_cuda.o
	${CC} -o $@ $^ ${LFLAGS_CUDA}

mixbench-cuda-bs: main-cuda.o mix_kernels_cuda-bs.o
	${CC} -o $@ $^ ${LFLAGS_CUDA}

mixbench-ocl: main-ocl.o mix_kernels_ocl.o
	${CC} -o $@ $^ ${LFLAGS_OCL}

main-cuda.o: main-cuda.cpp mix_kernels_cuda.h lcutil.h
	${CC} -c ${FLAGS_CUDA} $<

main-ocl.o: main-ocl.cpp mix_kernels_ocl.h loclutil.h
	${CC} -c ${FLAGS_OCL} $<

mix_kernels_cuda.o: mix_kernels_cuda.cu lcutil.h
	${NVCC} ${NVCODE} ${NVFLAGS} -DUNIX -c $< -o $@

mix_kernels_cuda-bs.o: mix_kernels_cuda.cu
	${NVCC} ${NVCODE} ${NVFLAGS} -DUNIX -DBLOCK_STRIDED -c $< -o $@

mix_kernels_ocl.o: mix_kernels_ocl.cpp loclutil.h
	${CC} -c ${FLAGS_OCL} $<

clean:
	\rm -f mixbench-cuda main-cuda.o mix_kernels_cuda.o mix_kernels_cuda-bs.o mixbench-cuda-bs mixbench-cuda-bs.o mixbench-ocl main-ocl.o mix_kernels_ocl.o

rebuild: clean mixbench-cuda mixbench-cuda-bs mixbench-ocl

