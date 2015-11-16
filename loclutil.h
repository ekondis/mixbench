/**
 * loclutil.h: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#ifndef _OCLUTIL_H_
#define _OCLUTIL_H_

#include <cstdio>
#include <cstdlib>
#include <CL/opencl.h>

#define OCL_SAFE_CALL(call) {                                                \
    cl_int err = call;                                                       \
    if( CL_SUCCESS != err) {                                                 \
        fprintf(stderr, "OpenCL error in file '%s' in line %i : Code %d.\n", \
                __FILE__, __LINE__, err );                                   \
        exit(EXIT_FAILURE);                                                  \
    } }

#define FRACTION_CEILING(numerator, denominator) ((numerator+denominator-1)/(denominator))

inline cl_device_id GetDeviceID(int index, FILE *fout){
	cl_uint cnt_platforms, cnt_device_ids;
	//cl_platform_id pid;
	cl_device_id device_selected = NULL;
	char dev_name[256];
	
	//OCL_SAFE_CALL( clGetPlatformIDs(1, &pid, NULL) );
	OCL_SAFE_CALL( clGetPlatformIDs(0, NULL, &cnt_platforms) );
	cl_platform_id *platform_ids = (cl_platform_id*)alloca(sizeof(cl_platform_id)*cnt_platforms);
	cl_device_id device_ids[256];
	OCL_SAFE_CALL( clGetPlatformIDs(cnt_platforms, platform_ids, NULL) );

	if( fout )
		fprintf(fout, "Available OpenCL devices:\n");
	int cur_dev_idx = 1;
	for(int i=0; i<(int)cnt_platforms; i++){
		OCL_SAFE_CALL( clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, 0, NULL, &cnt_device_ids) );
		//cl_device_id *device_ids = (cl_device_id*)alloca(sizeof(cl_device_id)*cnt_device_ids);
		OCL_SAFE_CALL( clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, cnt_device_ids, device_ids, NULL) );
		for(int d=0; d<(int)cnt_device_ids; d++){
			if( fout ){
				OCL_SAFE_CALL( clGetDeviceInfo(device_ids[d], CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL) );
				fprintf(fout, "%d. %s\n", cur_dev_idx, dev_name);
			}
			if( cur_dev_idx==index )
				device_selected = device_ids[d];
			cur_dev_idx++;
		}
	}
	return device_selected;
//	OCL_SAFE_CALL( clGetDeviceIDs(pid, CL_DEVICE_TYPE_ALL, 1, &did, NULL) );
//	return did;
}

// Print basic device information
inline void StoreDeviceInfo(cl_device_id devID, FILE *fout){
	char dev_name[256], dev_clver[256], dev_drv[256];
	cl_uint dev_freq, dev_cus;
	cl_ulong dev_cache, dev_gmem, dev_maxalloc;
	OCL_SAFE_CALL( clGetDeviceInfo (devID, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL) );
	OCL_SAFE_CALL( clGetDeviceInfo (devID, CL_DEVICE_VERSION, sizeof(dev_clver), dev_clver, NULL) );
	OCL_SAFE_CALL( clGetDeviceInfo (devID, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(dev_freq), &dev_freq, NULL) );
	OCL_SAFE_CALL( clGetDeviceInfo (devID, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(dev_cache), &dev_cache, NULL) );
	OCL_SAFE_CALL( clGetDeviceInfo (devID, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(dev_gmem), &dev_gmem, NULL) );
	OCL_SAFE_CALL( clGetDeviceInfo (devID, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(dev_maxalloc), &dev_maxalloc, NULL) );
	OCL_SAFE_CALL( clGetDeviceInfo (devID, CL_DRIVER_VERSION, sizeof(dev_drv), dev_drv, NULL) );
	OCL_SAFE_CALL( clGetDeviceInfo (devID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(dev_cus), &dev_cus, NULL) );
	fprintf(fout, "------------------------ Device specifications ------------------------\n");
	fprintf(fout, "Device:              %s\n", dev_name);
	fprintf(fout, "Driver version:      %s\n", dev_drv);
	fprintf(fout, "GPU clock rate:      %d MHz\n", dev_freq);
//	fprintf(fout, "WarpSize:            %d\n", deviceProp.warpSize);
	fprintf(fout, "Cache size:          %ld KB\n", dev_cache/1024);
	fprintf(fout, "Total global mem:    %d MB\n", (int)(dev_gmem/1024/1024));
	fprintf(fout, "Max allowed buffer:  %d MB\n", (int)(dev_maxalloc/1024/1024));
	fprintf(fout, "OpenCL version:      %s\n", dev_clver);
	fprintf(fout, "Total CUs:           %d\n", dev_cus);
	fprintf(fout, "-----------------------------------------------------------------------\n");
}

#endif
