/**
 * mix_kernels_cuda_ro.cu: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <stdio.h>
#include <math_constants.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
#include "lcutil.h"

#define ELEMENTS_PER_THREAD (8)
#define FUSION_DEGREE (4)

template<class T>
inline __device__ T conv_int(const int i){ return static_cast<T>(i); }

template<class T>
inline __device__ T mad(const T a, const T b, const T c){ return a*b+c; }

template<class T>
inline __device__ bool equal(const T a, const T b){ return a==b; }

#if __CUDA_ARCH__ >= 530
template<>
inline __device__ half2 conv_int(const int i){ return __half2half2( __int2half_rd(i) ); }
template<>
inline __device__ half2 mad(const half2 a, const half2 b, const half2 c){ return __hfma2(a, b, c)/*__hadd2(__hmul2(a, b), c)*/; }
template<>
inline __device__ bool equal(const half2 a, const half2 b){ return __hbeq2(a, b); }
#else
// a dummy implementations as a workaround
template<>
inline __device__ half2 conv_int(const int i){ return half2(); }
template<>
inline __device__ half2 mad(const half2 a, const half2 b, const half2 c){ return half2(); }
template<>
inline __device__ bool equal(const half2 a, const half2 b){ return false; }
#endif

template <class T, int blockdim, unsigned int granularity, unsigned int fusion_degree, unsigned int compute_iterations, bool TemperateUnroll>
__global__ void benchmark_func(T seed, T *g_data){
	const unsigned int blockSize = blockdim;
	const int stride = blockSize;
	int idx = blockIdx.x*blockSize*granularity + threadIdx.x;
	const int big_stride = gridDim.x*blockSize*granularity;

	T tmps[granularity];
	for(int k=0; k<fusion_degree; k++){
		#pragma unroll
		for(int j=0; j<granularity; j++){
			// Load elements (memory intensive part)
			tmps[j] = g_data[idx+j*stride+k*big_stride];
			// Perform computations (compute intensive part)
			#pragma unroll TemperateUnroll ? 4 : 128
			for(int i=0; i<compute_iterations; i++){
				tmps[j] = mad(tmps[j], tmps[j], seed);
			}
		}
		// Multiply add reduction
		T sum = conv_int<T>(0);
		#pragma unroll
		for(int j=0; j<granularity; j+=2)
			sum = mad(tmps[j], tmps[j+1], sum);
		// Dummy code
		if( equal(sum, conv_int<T>(-1)) ) // Designed so it never executes
			g_data[idx+k*big_stride] = sum;
	}
}

void initializeEvents(cudaEvent_t *start, cudaEvent_t *stop){
	CUDA_SAFE_CALL( cudaEventCreate(start) );
	CUDA_SAFE_CALL( cudaEventCreate(stop) );
	CUDA_SAFE_CALL( cudaEventRecord(*start, 0) );
}

float finalizeEvents(cudaEvent_t start, cudaEvent_t stop){
	CUDA_SAFE_CALL( cudaGetLastError() );
	CUDA_SAFE_CALL( cudaEventRecord(stop, 0) );
	CUDA_SAFE_CALL( cudaEventSynchronize(stop) );
	float kernel_time;
	CUDA_SAFE_CALL( cudaEventElapsedTime(&kernel_time, start, stop) );
	CUDA_SAFE_CALL( cudaEventDestroy(start) );
	CUDA_SAFE_CALL( cudaEventDestroy(stop) );
	return kernel_time;
}

void runbench_warmup(double *cd, long size){
	const long reduced_grid_size = size/(ELEMENTS_PER_THREAD)/128;
	const int BLOCK_SIZE = 256;
	const int TOTAL_REDUCED_BLOCKS = reduced_grid_size/BLOCK_SIZE;

	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimReducedGrid(TOTAL_REDUCED_BLOCKS, 1, 1);

	benchmark_func< short, BLOCK_SIZE, ELEMENTS_PER_THREAD, FUSION_DEGREE, 0, true ><<< dimReducedGrid, dimBlock >>>((short)1, (short*)cd);
	CUDA_SAFE_CALL( cudaGetLastError() );
	CUDA_SAFE_CALL( cudaDeviceSynchronize() );
}

int out_config = 1;

template<unsigned int compute_iterations>
void runbench(double *cd, long size, bool doHalfs){
	const long compute_grid_size = size/ELEMENTS_PER_THREAD/FUSION_DEGREE;
	const int BLOCK_SIZE = 256;
	const int TOTAL_BLOCKS = compute_grid_size/BLOCK_SIZE;
	const long long computations = (ELEMENTS_PER_THREAD*(long long)compute_grid_size+(2*ELEMENTS_PER_THREAD*compute_iterations)*(long long)compute_grid_size)*FUSION_DEGREE;
	const long long memoryoperations = size;

	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(TOTAL_BLOCKS, 1, 1);
	cudaEvent_t start, stop;

	initializeEvents(&start, &stop);
	benchmark_func< float, BLOCK_SIZE, ELEMENTS_PER_THREAD, FUSION_DEGREE, compute_iterations, false ><<< dimGrid, dimBlock >>>(1.0f, (float*)cd);
	float kernel_time_mad_sp = finalizeEvents(start, stop);

	initializeEvents(&start, &stop);
	benchmark_func< double, BLOCK_SIZE, ELEMENTS_PER_THREAD, FUSION_DEGREE, compute_iterations, false ><<< dimGrid, dimBlock >>>(1.0, cd);
	float kernel_time_mad_dp = finalizeEvents(start, stop);

	float kernel_time_mad_hp = 0.f;
	if( doHalfs ){
		initializeEvents(&start, &stop);
		half2 h_ones;
		*((int32_t*)&h_ones) = 15360 + (15360 << 16); // 1.0 as half
		benchmark_func< half2, BLOCK_SIZE, ELEMENTS_PER_THREAD, FUSION_DEGREE, compute_iterations, false ><<< dimGrid, dimBlock >>>(h_ones, (half2*)cd);
		kernel_time_mad_hp = finalizeEvents(start, stop);
	}

	initializeEvents(&start, &stop);
	benchmark_func< int, BLOCK_SIZE, ELEMENTS_PER_THREAD, FUSION_DEGREE, compute_iterations, true ><<< dimGrid, dimBlock >>>(1, (int*)cd);
	float kernel_time_mad_int = finalizeEvents(start, stop);

	printf("         %4d,   %8.3f,%8.2f,%8.2f,%7.2f,   %8.3f,%8.2f,%8.2f,%7.2f,   %8.3f,%8.2f,%8.2f,%7.2f,  %8.3f,%8.2f,%8.2f,%7.2f\n",
		compute_iterations,
		((double)computations)/((double)memoryoperations*sizeof(float)),
		kernel_time_mad_sp,
		((double)computations)/kernel_time_mad_sp*1000./(double)(1000*1000*1000),
		((double)memoryoperations*sizeof(float))/kernel_time_mad_sp*1000./(1000.*1000.*1000.),
		((double)computations)/((double)memoryoperations*sizeof(double)),
		kernel_time_mad_dp,
		((double)computations)/kernel_time_mad_dp*1000./(double)(1000*1000*1000),
		((double)memoryoperations*sizeof(double))/kernel_time_mad_dp*1000./(1000.*1000.*1000.),
		((double)2*computations)/((double)memoryoperations*sizeof(half2)),
		kernel_time_mad_hp,
		((double)2*computations)/kernel_time_mad_hp*1000./(double)(1000*1000*1000),
		((double)memoryoperations*sizeof(half2))/kernel_time_mad_hp*1000./(1000.*1000.*1000.),
		((double)computations)/((double)memoryoperations*sizeof(int)),
		kernel_time_mad_int,
		((double)computations)/kernel_time_mad_int*1000./(double)(1000*1000*1000),
		((double)memoryoperations*sizeof(int))/kernel_time_mad_int*1000./(1000.*1000.*1000.) );
}

extern "C" void mixbenchGPU(double *c, long size){
	const char *benchtype = "compute with global memory (block strided)";

	printf("Trade-off type:       %s\n", benchtype);
	printf("Elements per thread:  %d\n", ELEMENTS_PER_THREAD);
	printf("Thread fusion degree: %d\n", FUSION_DEGREE);
	double *cd;
	bool doHalfs = IsFP16Supported();
	if( !doHalfs )
		printf("Warning:              Half precision computations are not supported\n");

	CUDA_SAFE_CALL( cudaMalloc((void**)&cd, size*sizeof(double)) );

	// Copy data to device memory
	CUDA_SAFE_CALL( cudaMemset(cd, 0, size*sizeof(double)) );  // initialize to zeros

	// Synchronize in order to wait for memory operations to finish
	CUDA_SAFE_CALL( cudaDeviceSynchronize() );

	printf("----------------------------------------------------------------------------- CSV data -----------------------------------------------------------------------------\n");
	printf("Experiment ID, Single Precision ops,,,,              Double precision ops,,,,              Half precision ops,,,,                Integer operations,,, \n");
	printf("Compute iters, Flops/byte, ex.time,  GFLOPS, GB/sec, Flops/byte, ex.time,  GFLOPS, GB/sec, Flops/byte, ex.time,  GFLOPS, GB/sec, Iops/byte, ex.time,   GIOPS, GB/sec\n");

	runbench_warmup(cd, size);

	runbench<0>(cd, size, doHalfs);
	runbench<1>(cd, size, doHalfs);
	runbench<2>(cd, size, doHalfs);
	runbench<3>(cd, size, doHalfs);
	runbench<4>(cd, size, doHalfs);
	runbench<5>(cd, size, doHalfs);
	runbench<6>(cd, size, doHalfs);
	runbench<7>(cd, size, doHalfs);
	runbench<8>(cd, size, doHalfs);
	runbench<9>(cd, size, doHalfs);
	runbench<10>(cd, size, doHalfs);
	runbench<11>(cd, size, doHalfs);
	runbench<12>(cd, size, doHalfs);
	runbench<13>(cd, size, doHalfs);
	runbench<14>(cd, size, doHalfs);
	runbench<15>(cd, size, doHalfs);
	runbench<16>(cd, size, doHalfs);
	runbench<17>(cd, size, doHalfs);
	runbench<18>(cd, size, doHalfs);
	runbench<20>(cd, size, doHalfs);
	runbench<22>(cd, size, doHalfs);
	runbench<24>(cd, size, doHalfs);
	runbench<28>(cd, size, doHalfs);
	runbench<32>(cd, size, doHalfs);
	runbench<40>(cd, size, doHalfs);
	runbench<48>(cd, size, doHalfs);
	runbench<56>(cd, size, doHalfs);
	runbench<64>(cd, size, doHalfs);
	runbench<80>(cd, size, doHalfs);
	runbench<96>(cd, size, doHalfs);
	runbench<128>(cd, size, doHalfs);
	runbench<192>(cd, size, doHalfs);
	runbench<256>(cd, size, doHalfs);
	runbench<512>(cd, size, doHalfs);
	runbench<1024>(cd, size, doHalfs);

	printf("--------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");

	// Copy results back to host memory
	CUDA_SAFE_CALL( cudaMemcpy(c, cd, size*sizeof(double), cudaMemcpyDeviceToHost) );

	CUDA_SAFE_CALL( cudaFree(cd) );

	CUDA_SAFE_CALL( cudaDeviceReset() );
}
