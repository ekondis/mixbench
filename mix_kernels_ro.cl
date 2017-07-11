#ifdef ENABLE_DP
	#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
#ifdef ENABLE_HP
	#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

bool is_equal(const class_T a, const class_T b){
#ifdef ENABLE_HP
	return a.x==b.x && a.y==b.y;
#else
	return a==b;
#endif
}

__kernel __attribute__((reqd_work_group_size(blockdim, 1, 1)))
void benchmark_func(class_T seed, global class_T *g_data){
	const unsigned int blockSize = blockdim;
#ifdef BLOCK_STRIDED
	const int stride = blockSize;
	const int idx = get_group_id(0)*blockSize*ELEMENTS_PER_THREAD + get_local_id(0);
#else
	const int grid_size = blockSize * get_num_groups(0);
	const int stride = grid_size;
	const int idx = get_global_id(0);
#endif
	const int big_stride = get_num_groups(0)*blockSize*ELEMENTS_PER_THREAD;

	class_T tmps[ELEMENTS_PER_THREAD];
	for(int k=0; k<FUSION_DEGREE; k++){	
		#pragma unroll
		for(int j=0; j<ELEMENTS_PER_THREAD; j++){
			// Load elements (memory intensive part)
			tmps[j] = g_data[idx+j*stride+k*big_stride];
			// Perform computations (compute intensive part)
			for(int i=0; i<COMPUTE_ITERATIONS; i++){
				tmps[j] = tmps[j]*tmps[j]+seed;//tmps[(j+ELEMENTS_PER_THREAD/2)%ELEMENTS_PER_THREAD];
			}
		}
		// Multiply add reduction
		class_T sum = (class_T)0;
		#pragma unroll
		for(int j=0; j<ELEMENTS_PER_THREAD; j+=2)
			sum += tmps[j]*tmps[j+1];
		// Dummy code
		if( is_equal(sum, (class_T)-1) ) // Designed so it never executes
			g_data[idx+k*big_stride] = sum;
	}
}
