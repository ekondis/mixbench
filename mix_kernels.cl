#ifdef ENABLE_DP
	#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#define COMP_ITERATIONS (8192)
#define UNROLL_ITERATIONS (32)
#define UNROLLED_MEMORY_ACCESSES (UNROLL_ITERATIONS/2)
#define REGBLOCK_SIZE (16)

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
	  r7 = r0+(class_T)(17),
	  r8 = r0+(class_T)(19),
	  r9 = r0+(class_T)(23),
	  rA = r0+(class_T)(29),
	  rB = r0+(class_T)(31),
	  rC = r0+(class_T)(37),
	  rD = r0+(class_T)(41),
	  rE = r0+(class_T)(43),
	  rF = r0+(class_T)(47);

	for(int j=0; j<COMP_ITERATIONS; j+=UNROLL_ITERATIONS){
		#pragma unroll
		for(int i=0; i<UNROLL_ITERATIONS-memory_ratio; i++){
			r0 = r0 * r0 + r8;
			r1 = r1 * r1 + r9;
			r2 = r2 * r2 + rA;
			r3 = r3 * r3 + rB;
			r4 = r4 * r4 + rC;
			r5 = r5 * r5 + rD;
			r6 = r6 * r6 + rE;
			r7 = r7 * r7 + rF;
			r8 = r8 * r8 + r0;
			r9 = r9 * r9 + r1;
			rA = rA * rA + r2;
			rB = rB * rB + r3;
			rC = rC * rC + r4;
			rD = rD * rD + r5;
			rE = rE * rE + r6;
			rF = rF * rF + r7;
		}
		bool do_write = true;
		int reg_idx = 0;
		#pragma unroll
		for(int i=UNROLL_ITERATIONS-memory_ratio; i<UNROLL_ITERATIONS; i++){
			// Each iteration maps to one memory operation
			class_T* const r = reg_idx==0 ? &r0 : (reg_idx==1 ? &r1 : (reg_idx==2 ? &r2 : (reg_idx==3 ? &r3 : (reg_idx==4 ? &r4 : (reg_idx==5 ? &r5 : (reg_idx==6 ? &r6 :
				   (reg_idx==7 ? &r7 : (reg_idx==8 ? &r8 : (reg_idx==9 ? &r9 : (reg_idx==10 ? &rA : (reg_idx==11 ? &rB : (reg_idx==12 ? &rC : (reg_idx==13 ? &rD : (reg_idx==14 ? &rE : &rF))))))))
				   ))))));
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
	if( (r0==(class_T)INFINITY) && (r1==(class_T)INFINITY) && (r2==(class_T)INFINITY) && (r3==(class_T)INFINITY) &&
	  (r4==(class_T)INFINITY) && (r5==(class_T)INFINITY) && (r6==(class_T)INFINITY) && (r7==(class_T)INFINITY) &&
	  (r8==(class_T)INFINITY) && (r9==(class_T)INFINITY) && (rA==(class_T)INFINITY) && (rB==(class_T)INFINITY) &&
	  (rC==(class_T)INFINITY) && (rD==(class_T)INFINITY) && (rE==(class_T)INFINITY) && (rF==(class_T)INFINITY) ){ // extremely unlikely to happen
		g_data[0] = r0+r1+r2+r3+r4+r5+r6+r7+r8+r9+rA+rB+rC+rD+rE+rF;
	}
}
