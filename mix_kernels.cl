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

#define COMP_ITERATIONS (8192)
#define UNROLL_ITERATIONS (32)
#define UNROLLED_MEMORY_ACCESSES (UNROLL_ITERATIONS/2)
#define REGBLOCK_SIZE (8)

__kernel __attribute__((reqd_work_group_size(blockdim, 1, 1)))
void benchmark_func(class_T seed, global volatile class_T *g_data){
#ifdef BLOCK_STRIDED
	const int index_stride = blockdim;
	const int index_base = get_group_id(0) * blockdim * UNROLLED_MEMORY_ACCESSES + get_local_id(0);
#else
	const int grid_size = blockdim * get_num_groups(0);
	const int globaltid = get_global_id(0);
	const int index_stride = grid_size;
	const int index_base = globaltid;
#endif
	const int halfarraysize = get_num_groups(0)*blockdim*UNROLLED_MEMORY_ACCESSES;
	const int offset_slips = 1+UNROLLED_MEMORY_ACCESSES-((memory_ratio+1)/2);
	const int array_index_bound = index_base+offset_slips*index_stride;
	const int initial_index_range = memory_ratio>0 ? UNROLLED_MEMORY_ACCESSES % ((memory_ratio+1)/2) : 1;
	int initial_index_factor = 0;
	global volatile class_T *data = g_data;

	int array_index = index_base;
	class_T r0 = seed + get_global_id(0),
	  r1 = r0+(class_T)(2),
	  r2 = r0+(class_T)(3),
	  r3 = r0+(class_T)(5),
	  r4 = r0+(class_T)(7),
	  r5 = r0+(class_T)(11),
	  r6 = r0+(class_T)(13),
	  r7 = r0+(class_T)(17);

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
			class_T* const r = reg_idx==0 ? &r0 : (reg_idx==1 ? &r1 : (reg_idx==2 ? &r2 : (reg_idx==3 ? &r3 : (reg_idx==4 ? &r4 : (reg_idx==5 ? &r5 : (reg_idx==6 ? &r6 : &r7))))));
			if( do_write )
				data[ array_index+halfarraysize ] = *r;
			else {
				*r = data[ array_index ];
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
	if( is_equal(r0, (class_T)INFINITY) && is_equal(r1, (class_T)INFINITY) && is_equal(r2, (class_T)INFINITY) && is_equal(r3, (class_T)INFINITY) &&
	    is_equal(r4, (class_T)INFINITY) && is_equal(r5, (class_T)INFINITY) && is_equal(r6, (class_T)INFINITY) && is_equal(r7, (class_T)INFINITY) ){ // extremely unlikely to happen
		g_data[0] = r0+r1+r2+r3+r4+r5+r6+r7;
	}
}
