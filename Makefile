CUDA_INSTALL_PATH = /usr/local/cuda
#OCL_INSTALL_PATH = /opt/AMDAPPSDK
OCL_INSTALL_PATH = ${CUDA_INSTALL_PATH}
CUDA_INC_PATH = ${CUDA_INSTALL_PATH}/include
CUDA_LIB_PATH = ${CUDA_INSTALL_PATH}/lib64
OCL_INC_PATH = ${OCL_INSTALL_PATH}/include
OCL_LIB_PATH = ${OCL_INSTALL_PATH}/lib/x86_64
CC = g++
OPTFLAG = -O2
NVCC = ${CUDA_INSTALL_PATH}/bin/nvcc
FLAGS_CUDA = ${OPTFLAG} -I${CUDA_INC_PATH} -Wall
FLAGS_OCL = ${OPTFLAG} -I${OCL_INC_PATH} -Wall
NVFLAGS = -O2 -I${CUDA_INC_PATH} --compiler-options -fno-strict-aliasing --ptxas-options=-v -Xptxas -dlcm=cg
BITS = $(shell getconf LONG_BIT)
LFLAGS_CUDA = -L${CUDA_LIB_PATH} -lm -lstdc++ -lcudart -lrt
LFLAGS_OCL = -L${OCL_LIB_PATH} -lm -lstdc++ -lOpenCL -lrt
NVCODE = -gencode=arch=compute_20,code=\"compute_20\"
#NVCODE = -gencode=arch=compute_52,code=\"compute_52\" -gencode=arch=compute_30,code=\"compute_30\" -gencode=arch=compute_20,code=\"compute_20\"
#NVCODE = -gencode=arch=compute_30,code=\"compute_30\"

.PHONY: all

ifdef CUDA_INSTALL_PATH
# build both cuda and opencl executables
all: mixbench-cuda mixbench-cuda-ro mixbench-ocl
else
# build opencl only executable
all: mixbench-ocl
endif

mixbench-cuda: main-cuda.o mix_kernels_cuda.o
	${CC} -o $@ $^ ${LFLAGS_CUDA}

mixbench-cuda-ro: main-cuda-ro.o mix_kernels_cuda_ro.o
	${CC} -o $@ $^ ${LFLAGS_CUDA}

mixbench-ocl: main-ocl.o mix_kernels_ocl.o
	${CC} -o $@ $^ ${LFLAGS_OCL}

main-cuda.o: main-cuda.cpp mix_kernels_cuda.h lcutil.h
	${CC} -c ${FLAGS_CUDA} $< -o $@
 
main-cuda-ro.o: main-cuda.cpp mix_kernels_cuda.h lcutil.h
	${CC} -c ${FLAGS_CUDA} -DREADONLY $< -o $@

main-ocl.o: main-ocl.cpp mix_kernels_ocl.h loclutil.h
	${CC} -c ${FLAGS_OCL} $< -o $@

mix_kernels_cuda.o: mix_kernels_cuda.cu mix_kernels_cuda.h lcutil.h
	${NVCC} ${NVCODE} ${NVFLAGS} -DUNIX -c $< -o $@

mix_kernels_cuda_ro.o: mix_kernels_cuda_ro.cu mix_kernels_cuda.h lcutil.h
	${NVCC} ${NVCODE} ${NVFLAGS} -DUNIX -c $< -o $@

mix_kernels_ocl.o: mix_kernels_ocl.cpp mix_kernels_ocl.h loclutil.h
	${CC} -c ${FLAGS_OCL} $< -o $@

clean:
	\rm -f mixbench-cuda main-cuda.o mix_kernels_cuda.o main-cuda-ro.o mixbench-cuda-ro mix_kernels_cuda_ro.o mixbench-ocl main-ocl.o mix_kernels_ocl.o

rebuild: clean mixbench-cuda mixbench-cuda-ro mixbench-ocl

