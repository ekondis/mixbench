CUDA_INSTALL_PATH = /usr/local/cuda
#OCL_INSTALL_PATH = /opt/AMDAPPSDK
#OCL_INSTALL_PATH = /opt/rocm/opencl
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
OS := $(shell uname)
ifeq ($(OS),Darwin)
    LFLAGS_OCL = -lm -lstdc++ -framework OpenCL
else
    LFLAGS_OCL = -L${OCL_LIB_PATH} -lm -lstdc++ -lOpenCL -lrt
endif
NVCODE = -gencode=arch=compute_60,code=\"compute_60\" -gencode=arch=compute_30,code=\"compute_30\"

ifdef HIP_PATH
    HIPCC=$(HIP_PATH)/bin/hipcc
    HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --compiler)
    ifeq (${HIP_PLATFORM}, nvcc)
        HIPCC_FLAGS=${NVCODE} ${NVFLAGS}
        LFLAGS_HIP =${LFLAGS_CUDA}
    else
        HIPCC_FLAGS=${OPTFLAG} -I${HIP_PATH}/hip/include
        LFLAGS_HIP =${OPTFLAG}
    endif
endif

.PHONY: all rebuild clean

ifdef CUDA_INSTALL_PATH
    ifdef HIP_PLATFORM
        # build hip, cuda and opencl executables
        all: mixbench-hip-alt mixbench-hip-ro mixbench-cuda-alt mixbench-cuda-ro mixbench-ocl-alt mixbench-ocl-ro
    else
        # build both cuda and opencl executables
        all: mixbench-cuda-alt mixbench-cuda-ro mixbench-ocl-alt mixbench-ocl-ro
    endif
else
    ifdef HIP_PLATFORM
        # build hip only executable
        all: mixbench-hip-alt mixbench-hip-ro
    else
        # build opencl only executable
        all: mixbench-ocl-alt mixbench-ocl-ro
    endif
endif

mixbench-cuda-alt: main-cuda.o mix_kernels_cuda.o
	${CC} -o $@ $^ ${LFLAGS_CUDA}

mixbench-cuda-ro: main-cuda-ro.o mix_kernels_cuda_ro.o
	${CC} -o $@ $^ ${LFLAGS_CUDA}

mixbench-ocl-alt: main-ocl.o mix_kernels_ocl.o
	${CC} -o $@ $^ ${LFLAGS_OCL}

mixbench-ocl-ro: main-ocl-ro.o mix_kernels_ocl_ro.o
	${CC} -o $@ $^ ${LFLAGS_OCL}

mixbench-hip-alt: main-hip.o mix_kernels_hip.o
	${HIPCC} ${LFLAGS_HIP} -o $@ $^

mixbench-hip-ro: main-hip-ro.o mix_kernels_hip-ro.o
	${HIPCC} ${LFLAGS_HIP} -o $@ $^

main-cuda.o: main-cuda.cpp mix_kernels_cuda.h lcutil.h version_info.h
	${CC} -c ${FLAGS_CUDA} $< -o $@

main-cuda-ro.o: main-cuda.cpp mix_kernels_cuda.h lcutil.h version_info.h
	${CC} -c ${FLAGS_CUDA} -DREADONLY $< -o $@

main-ocl.o: main-ocl.cpp mix_kernels_ocl.h loclutil.h version_info.h
	${CC} -c ${FLAGS_OCL} $< -o $@

main-ocl-ro.o: main-ocl.cpp mix_kernels_ocl.h loclutil.h version_info.h
	${CC} -c ${FLAGS_OCL} -DREADONLY $< -o $@

mix_kernels_cuda.o: mix_kernels_cuda.cu mix_kernels_cuda.h lcutil.h
	${NVCC} ${NVCODE} ${NVFLAGS} -DUNIX -c $< -o $@

mix_kernels_cuda_ro.o: mix_kernels_cuda_ro.cu mix_kernels_cuda.h lcutil.h
	${NVCC} ${NVCODE} ${NVFLAGS} -DUNIX -c $< -o $@

mix_kernels_ocl.o: mix_kernels_ocl.cpp mix_kernels_ocl.h loclutil.h
	${CC} -c ${FLAGS_OCL} $< -o $@

mix_kernels_ocl_ro.o: mix_kernels_ocl_ro.cpp mix_kernels_ocl.h loclutil.h
	${CC} -c ${FLAGS_OCL} $< -o $@

#HIP
main-hip.o: main-hip.cpp mix_kernels_hip.h lhiputil.h version_info.h
	${HIPCC} -c ${HIPCC_FLAGS} $<

mix_kernels_hip.o: mix_kernels_hip.cpp lhiputil.h
	${HIPCC} ${HIPCC_FLAGS} -DUNIX -c $< -o $@

main-hip-ro.o: main-hip-ro.cpp mix_kernels_hip.h lhiputil.h version_info.h
	${HIPCC} -c ${HIPCC_FLAGS} -DREADONLY $<

mix_kernels_hip-ro.o: mix_kernels_hip_ro.cpp lhiputil.h
	${HIPCC} ${HIPCC_FLAGS} -DUNIX -c $< -o $@

version_info.h:
	echo '#define VERSION_INFO "'`./query_version.sh`'"' >$@

clean:
	\rm -f mixbench-cuda-alt mixbench-cuda-ro mixbench-ocl-alt mixbench-ocl-ro mixbench-hip-alt mixbench-hip-ro *.o version_info.h

rebuild: clean mixbench-cuda-alt mixbench-cuda-ro mixbench-ocl-alt mixbench-ocl-ro mixbench-hip-alt mixbench-hip-ro
