#define class_T float
#define blockdim 256
#define memory_ratio 16
#define griddim 16384
#define BLOCK_STRIDED

#define COMP_ITERATIONS (8192)
#define UNROLL_ITERATIONS (32)
#define UNROLLED_MEMORY_ACCESSES (UNROLL_ITERATIONS/2)

__kernel void benchmark_func(class_T seed, global volatile class_T *g_data){
#ifdef BLOCK_STRIDED
	const int index_stride = blockdim;
	const int index_base = get_group_id(0) * blockdim * UNROLLED_MEMORY_ACCESSES + get_local_id(0);
#else
	const int grid_size = blockdim * (griddim == 0 ? get_num_groups(0) : griddim);
	const int globaltid = get_global_id(0);
	const int index_stride = grid_size;
	const int index_base = globaltid;
#endif
	const int halfarraysize = get_num_groups(0)*blockdim*UNROLLED_MEMORY_ACCESSES;
	const int offset_slips = 1+UNROLLED_MEMORY_ACCESSES-((memory_ratio+1)/2);
	const int array_index_bound = index_base+offset_slips*index_stride;
	global volatile class_T *data = g_data;

	int array_index = index_base;
	class_T r0 = seed,
	  r1 = r0+(class_T)(31),
	  r2 = r0+(class_T)(37),
	  r3 = r0+(class_T)(41);

	for(int j=0; j<COMP_ITERATIONS; j+=UNROLL_ITERATIONS){
		#pragma unroll
		for(int i=0; i<UNROLL_ITERATIONS-memory_ratio; i++){
			// Each iteration maps to floating point 8 operations (4 multiplies + 4 additions)
			r0 = r0 * r0 + r1;
			r1 = r1 * r1 + r2;
			r2 = r2 * r2 + r3;
			r3 = r3 * r3 + r0;
		}
		bool do_write = true;
		int reg_idx = 0;
		#pragma unroll
		for(int i=UNROLL_ITERATIONS-memory_ratio; i<UNROLL_ITERATIONS; i++){
			// Each iteration maps to one memory operation
			class_T* const r = reg_idx==0 ? &r0 : (reg_idx==1 ? &r1 : (reg_idx==2 ? &r2 : &r3));
			if( do_write )
				data[ array_index+halfarraysize ] = *r;
			else {
				*r = data[ array_index ];
				if( ++reg_idx>3 )
					reg_idx = 0;
				array_index += index_stride;
			}
			do_write = !do_write;
		}
		if( array_index >= array_index_bound )
			array_index = index_base;
	}
	if( (r0==(class_T)INFINITY) && (r1==(class_T)INFINITY) && (r2==(class_T)INFINITY) && (r3==(class_T)INFINITY) ){ // extremely unlikely to happen
		g_data[0] = r0+r1+r2+r3; 
	}
}
