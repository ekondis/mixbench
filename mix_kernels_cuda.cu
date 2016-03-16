/**
 * mix_kernels_cuda.cu: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <stdio.h>
#include <math_constants.h>
#include "lcutil.h"

#define COMP_ITERATIONS (8192)
#define UNROLL_ITERATIONS (32)
#define REGBLOCK_SIZE (8)

#define UNROLLED_MEMORY_ACCESSES (UNROLL_ITERATIONS/2)

template <class T, int blockdim, int memory_ratio>
__global__ void benchmark_func(T seed, volatile T *g_data){
	const int index_stride = blockdim;
	const int index_base = blockIdx.x*blockdim*UNROLLED_MEMORY_ACCESSES + threadIdx.x;

	const int halfarraysize = gridDim.x*blockdim*UNROLLED_MEMORY_ACCESSES;
	const int offset_slips = 1+UNROLLED_MEMORY_ACCESSES-((memory_ratio+1)/2);
	const int array_index_bound = index_base+offset_slips*index_stride;
	const int initial_index_range = memory_ratio>0 ? UNROLLED_MEMORY_ACCESSES % ((memory_ratio+1)/2) : 1;
	int initial_index_factor = 0;
	volatile T *data = g_data;

	int array_index = index_base;
	T r0 = seed + blockIdx.x * blockdim + threadIdx.x,
	  r1 = r0+(T)(2),
	  r2 = r0+(T)(3),
	  r3 = r0+(T)(5),
	  r4 = r0+(T)(7),
	  r5 = r0+(T)(11),
	  r6 = r0+(T)(13),
	  r7 = r0+(T)(17);

	for(int j=0; j<COMP_ITERATIONS; j+=UNROLL_ITERATIONS){
		#pragma unroll
		for(int i=0; i<UNROLL_ITERATIONS-memory_ratio; i++){
			r0 = r0 * r0 + r4;
			r1 = r1 * r1 + r5;
			r2 = r2 * r2 + r6;
			r3 = r3 * r3 + r7;
			r4 = r4 * r4 + r0;
			r5 = r5 * r5 + r1;
			r6 = r6 * r6 + r2;
			r7 = r7 * r7 + r3;
		}
		bool do_write = true;
		int reg_idx = 0;
		#pragma unroll
		for(int i=UNROLL_ITERATIONS-memory_ratio; i<UNROLL_ITERATIONS; i++){
			// Each iteration maps to one memory operation
			T& r = reg_idx==0 ? r0 : (reg_idx==1 ? r1 : (reg_idx==2 ? r2 : (reg_idx==3 ? r3 : (reg_idx==4 ? r4 : (reg_idx==5 ? r5 : (reg_idx==6 ? r6 : r7))))));
			if( do_write )
				data[ array_index+halfarraysize ] = r;
			else {
				r = data[ array_index ];
				if( ++reg_idx>=REGBLOCK_SIZE )
					reg_idx = 0;
				array_index += index_stride;
			}
			do_write = !do_write;
		}
		if( array_index >= array_index_bound ){
			if( ++initial_index_factor > initial_index_range)
				initial_index_factor = 0;
			array_index = index_base + initial_index_factor*index_stride;
		}
	}
	if( (r0==(T)CUDART_INF) && (r1==(T)CUDART_INF) && (r2==(T)CUDART_INF) && (r3==(T)CUDART_INF) &&
	    (r4==(T)CUDART_INF) && (r5==(T)CUDART_INF) && (r6==(T)CUDART_INF) && (r7==(T)CUDART_INF) ){ // extremely unlikely to happen
		g_data[0] = r0+r1+r2+r3+r4+r5+r6+r7;
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
	const long reduced_grid_size = size/(UNROLLED_MEMORY_ACCESSES)/32;
	const int BLOCK_SIZE = 256;
	const int TOTAL_REDUCED_BLOCKS = reduced_grid_size/BLOCK_SIZE;

	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimReducedGrid(TOTAL_REDUCED_BLOCKS, 1, 1);

	benchmark_func< short, BLOCK_SIZE, 0 ><<< dimReducedGrid, dimBlock >>>((short)1, (short*)cd);
	CUDA_SAFE_CALL( cudaGetLastError() );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

template<int memory_ratio>
void runbench(double *cd, long size){
	if( memory_ratio>UNROLL_ITERATIONS ){
		fprintf(stderr, "ERROR: memory_ratio exceeds UNROLL_ITERATIONS\n");
		exit(1);
	}
		
	const long compute_grid_size = size/(UNROLLED_MEMORY_ACCESSES)/2;
	const int BLOCK_SIZE = 256;
	const int TOTAL_BLOCKS = compute_grid_size/BLOCK_SIZE;
	const long long computations = 2*(long long)(COMP_ITERATIONS)*REGBLOCK_SIZE*compute_grid_size;
	const long long memoryoperations = (long long)(COMP_ITERATIONS)*compute_grid_size;

	dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid(TOTAL_BLOCKS, 1, 1);
	cudaEvent_t start, stop;

	initializeEvents(&start, &stop);
	benchmark_func< float, BLOCK_SIZE, memory_ratio ><<< dimGrid, dimBlock >>>(1.0f, (float*)cd);
	float kernel_time_mad_sp = finalizeEvents(start, stop);

	initializeEvents(&start, &stop);
	benchmark_func< double, BLOCK_SIZE, memory_ratio ><<< dimGrid, dimBlock >>>(1.0, cd);
	float kernel_time_mad_dp = finalizeEvents(start, stop);

	initializeEvents(&start, &stop);
	benchmark_func< int, BLOCK_SIZE, memory_ratio ><<< dimGrid, dimBlock >>>(1, (int*)cd);
	float kernel_time_mad_int = finalizeEvents(start, stop);

	const double memaccesses_ratio = (double)(memory_ratio)/UNROLL_ITERATIONS;
	const double computations_ratio = 1.0-memaccesses_ratio;

	printf("         %4d,  %8.3f,%8.2f,%8.2f,%7.2f,   %8.3f,%8.2f,%8.2f,%7.2f,  %8.3f,%8.2f,%8.2f,%7.2f\n",
		UNROLL_ITERATIONS-memory_ratio,
		(computations_ratio*(double)computations)/(memaccesses_ratio*(double)memoryoperations*sizeof(float)),
		kernel_time_mad_sp,
		(computations_ratio*(double)computations)/kernel_time_mad_sp*1000./(double)(1000*1000*1000),
		(memaccesses_ratio*(double)memoryoperations*sizeof(float))/kernel_time_mad_sp*1000./(1000.*1000.*1000.),
		(computations_ratio*(double)computations)/(memaccesses_ratio*(double)memoryoperations*sizeof(double)),
		kernel_time_mad_dp,
		(computations_ratio*(double)computations)/kernel_time_mad_dp*1000./(double)(1000*1000*1000),
		(memaccesses_ratio*(double)memoryoperations*sizeof(double))/kernel_time_mad_dp*1000./(1000.*1000.*1000.),
		(computations_ratio*(double)computations)/(memaccesses_ratio*(double)memoryoperations*sizeof(int)),
		kernel_time_mad_int,
		(computations_ratio*(double)computations)/kernel_time_mad_int*1000./(double)(1000*1000*1000),
		(memaccesses_ratio*(double)memoryoperations*sizeof(int))/kernel_time_mad_int*1000./(1000.*1000.*1000.) );
}

extern "C" void mixbenchGPU(double *c, long size){
	const char *benchtype = "compute with global memory (block strided)";

	printf("Trade-off type:%s\n", benchtype);
	double *cd;

	CUDA_SAFE_CALL( cudaMalloc((void**)&cd, size*sizeof(double)) );

	// Copy data to device memory
	CUDA_SAFE_CALL( cudaMemset(cd, 0, size*sizeof(double)) );  // initialize to zeros

	// Synchronize in order to wait for memory operations to finish
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	printf("---------------------------------------------------------- CSV data ----------------------------------------------------------\n");
	printf("Experiment ID, Single Precision ops,,,,              Double precision ops,,,,              Integer operations,,, \n");
	printf("Compute iters, Flops/byte, ex.time,  GFLOPS, GB/sec, Flops/byte, ex.time,  GFLOPS, GB/sec, Iops/byte, ex.time,   GIOPS, GB/sec\n");

	runbench_warmup(cd, size);

	runbench<32>(cd, size);
	runbench<31>(cd, size);
	runbench<30>(cd, size);
	runbench<29>(cd, size);
	runbench<28>(cd, size);
	runbench<27>(cd, size);
	runbench<26>(cd, size);
	runbench<25>(cd, size);
	runbench<24>(cd, size);
	runbench<23>(cd, size);
	runbench<22>(cd, size);
	runbench<21>(cd, size);
	runbench<20>(cd, size);
	runbench<19>(cd, size);
	runbench<18>(cd, size);
	runbench<17>(cd, size);
	runbench<16>(cd, size);
	runbench<15>(cd, size);
	runbench<14>(cd, size);
	runbench<13>(cd, size);
	runbench<12>(cd, size);
	runbench<11>(cd, size);
	runbench<10>(cd, size);
	runbench<9>(cd, size);
	runbench<8>(cd, size);
	runbench<7>(cd, size);
	runbench<6>(cd, size);
	runbench<5>(cd, size);
	runbench<4>(cd, size);
	runbench<3>(cd, size);
	runbench<2>(cd, size);
	runbench<1>(cd, size);
	runbench<0>(cd, size);

	printf("------------------------------------------------------------------------------------------------------------------------------\n");

	// Copy results back to host memory
	CUDA_SAFE_CALL( cudaMemcpy(c, cd, size*sizeof(double), cudaMemcpyDeviceToHost) );

	CUDA_SAFE_CALL( cudaFree(cd) );

	CUDA_SAFE_CALL( cudaDeviceReset() );
}
