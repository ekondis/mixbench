/**
 * loclutil.h: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#ifndef _OCLUTIL_H_
#define _OCLUTIL_H_

#include <cstdio>
#include <cstdlib>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/opencl.h>

#if defined(_MSC_VER)
#include <malloc.h>
#define alloca _alloca
#endif

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
	cl_device_id device_selected = NULL;
	char dev_name[256], plat_name[256];

	OCL_SAFE_CALL( clGetPlatformIDs(0, NULL, &cnt_platforms) );
	cl_platform_id *platform_ids = (cl_platform_id*)alloca(sizeof(cl_platform_id)*cnt_platforms);
	cl_device_id device_ids[256];
	OCL_SAFE_CALL( clGetPlatformIDs(cnt_platforms, platform_ids, NULL) );

	if( fout )
		fprintf(fout, "Available OpenCL devices:\n");
	int cur_dev_idx = 1;
	for(int i=0; i<(int)cnt_platforms; i++){
		size_t sz_name_len;
		OCL_SAFE_CALL( clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, 0, NULL, &sz_name_len) );
		sz_name_len = sz_name_len>sizeof(plat_name) ? sizeof(plat_name) : sz_name_len;
		OCL_SAFE_CALL( clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, sz_name_len, plat_name, NULL) );

		OCL_SAFE_CALL( clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, 0, NULL, &cnt_device_ids) );
		OCL_SAFE_CALL( clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, cnt_device_ids, device_ids, NULL) );
		for(int d=0; d<(int)cnt_device_ids; d++){
			if( fout ){
				OCL_SAFE_CALL( clGetDeviceInfo(device_ids[d], CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL) );
				fprintf(fout, "  %d. %s/%s\n", cur_dev_idx, dev_name, plat_name);
			}
			if( cur_dev_idx==index )
				device_selected = device_ids[d];
			cur_dev_idx++;
		}
	}
	return device_selected;
}

// Print basic device information
inline void StoreDeviceInfo(cl_device_id devID, FILE *fout){
	char dev_platform[256], dev_name[256], dev_vendor[256], dev_clver[256], dev_drv[256];
	cl_uint dev_freq, dev_cus, dev_addrbits;
	cl_ulong dev_gmem, dev_maxalloc;
	cl_platform_id dev_platform_id;
	OCL_SAFE_CALL( clGetDeviceInfo (devID, CL_DEVICE_PLATFORM, sizeof(dev_platform_id), &dev_platform_id, NULL) );
	OCL_SAFE_CALL( clGetPlatformInfo(dev_platform_id, CL_PLATFORM_NAME, sizeof(dev_platform), dev_platform, NULL) );
	OCL_SAFE_CALL( clGetDeviceInfo (devID, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL) );
	OCL_SAFE_CALL( clGetDeviceInfo (devID, CL_DEVICE_VENDOR, sizeof(dev_vendor), dev_vendor, NULL) );
	OCL_SAFE_CALL( clGetDeviceInfo (devID, CL_DEVICE_VERSION, sizeof(dev_clver), dev_clver, NULL) );
	OCL_SAFE_CALL( clGetDeviceInfo (devID, CL_DEVICE_ADDRESS_BITS, sizeof(dev_addrbits), &dev_addrbits, NULL) );
	OCL_SAFE_CALL( clGetDeviceInfo (devID, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(dev_freq), &dev_freq, NULL) );
	OCL_SAFE_CALL( clGetDeviceInfo (devID, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(dev_gmem), &dev_gmem, NULL) );
	OCL_SAFE_CALL( clGetDeviceInfo (devID, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(dev_maxalloc), &dev_maxalloc, NULL) );
	OCL_SAFE_CALL( clGetDeviceInfo (devID, CL_DRIVER_VERSION, sizeof(dev_drv), dev_drv, NULL) );
	OCL_SAFE_CALL( clGetDeviceInfo (devID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(dev_cus), &dev_cus, NULL) );
	fprintf(fout, "------------------------ Device specifications ------------------------\n");
	fprintf(fout, "Platform:            %s\n", dev_platform);
	fprintf(fout, "Device:              %s/%s\n", dev_name, dev_vendor);
	fprintf(fout, "Driver version:      %s\n", dev_drv);
	fprintf(fout, "Address bits:        %d\n", dev_addrbits);
	fprintf(fout, "GPU clock rate:      %d MHz\n", dev_freq);
	fprintf(fout, "Total global mem:    %d MB\n", (int)(dev_gmem/1024/1024));
	fprintf(fout, "Max allowed buffer:  %d MB\n", (int)(dev_maxalloc/1024/1024));
	fprintf(fout, "OpenCL version:      %s\n", dev_clver);
	fprintf(fout, "Total CUs:           %d\n", dev_cus);
	fprintf(fout, "-----------------------------------------------------------------------\n");
}

inline size_t GetMaxDeviceWGSize(cl_device_id devID){
	size_t wgsize;
	OCL_SAFE_CALL( clGetDeviceInfo (devID, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(wgsize), &wgsize, NULL) );
	return wgsize;
}

#endif
