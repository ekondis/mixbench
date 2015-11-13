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

// Print basic device information
static void StoreDeviceInfo(cl_device_id devID, FILE *fout){
	char dev_name[256], dev_clver[256], dev_drv[256];
	cl_uint dev_freq, dev_cus;
	cl_ulong dev_cache, dev_gmem;
	OCL_SAFE_CALL( clGetDeviceInfo (devID, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL) );
	OCL_SAFE_CALL( clGetDeviceInfo (devID, CL_DEVICE_VERSION, sizeof(dev_clver), dev_clver, NULL) );
	OCL_SAFE_CALL( clGetDeviceInfo (devID, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(dev_freq), &dev_freq, NULL) );
	OCL_SAFE_CALL( clGetDeviceInfo (devID, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(dev_cache), &dev_cache, NULL) );
	OCL_SAFE_CALL( clGetDeviceInfo (devID, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(dev_gmem), &dev_gmem, NULL) );
	OCL_SAFE_CALL( clGetDeviceInfo (devID, CL_DRIVER_VERSION, sizeof(dev_drv), dev_drv, NULL) );
	OCL_SAFE_CALL( clGetDeviceInfo (devID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(dev_cus), &dev_cus, NULL) );
	fprintf(fout, "------------------------ Device specifications ------------------------\n");
	fprintf(fout, "Device:              %s\n", dev_name);
	fprintf(fout, "Driver version:      %s\n", dev_drv);
	fprintf(fout, "GPU clock rate:      %d MHz\n", dev_freq);
//	fprintf(fout, "WarpSize:            %d\n", deviceProp.warpSize);
	fprintf(fout, "Cache size:          %d KB\n", dev_cache/1024);
	fprintf(fout, "Total global mem:    %d MB\n", (int)(dev_gmem/1024/1024));
	fprintf(fout, "OpenCL version:      %s\n", dev_clver);
	fprintf(fout, "Total CUs:           %d\n", dev_cus);
	fprintf(fout, "-----------------------------------------------------------------------\n");
}

#endif
