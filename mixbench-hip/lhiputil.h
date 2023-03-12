/**
 * lhiputil.h: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#ifndef _HIPUTIL_H_
#define _HIPUTIL_H_

#include <stdio.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#define HIP_SAFE_CALL( call) {                                    \
    hipError_t err = call;                                                    \
    if( hipSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, hipGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }

#define FRACTION_CEILING(numerator, denominator) ((numerator+denominator-1)/(denominator))

static inline int _ConvertSMVer2Cores(int major, int minor){
#ifdef __HIP_PLATFORM_HCC__
	return 64;
#else
	switch(major){
		case 1:  return 8;
		case 2:  switch(minor){
			case 1:  return 48;
			default: return 32;
		}
		case 3:  return 192;
		default: return 128;
	}
#endif
}

static inline void GetDevicePeakInfo(double *aGIPS, double *aGBPS, hipDeviceProp_t *aDeviceProp = NULL){
	hipDeviceProp_t deviceProp;
	int current_device;
	if( aDeviceProp )
		deviceProp = *aDeviceProp;
	else{
		HIP_SAFE_CALL( hipGetDevice(&current_device) );
		HIP_SAFE_CALL( hipGetDeviceProperties(&deviceProp, current_device) );
	}
	const int TotalSPs = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor)*deviceProp.multiProcessorCount;
	*aGIPS = 1000.0 * deviceProp.clockRate * TotalSPs / (1000.0 * 1000.0 * 1000.0);  // Giga instructions/sec
	*aGBPS = 2.0 * (double)deviceProp.memoryClockRate * 1000.0 * (double)deviceProp.memoryBusWidth / 8.0;
}

static inline hipDeviceProp_t GetDeviceProperties(void){
	hipDeviceProp_t deviceProp;
	int current_device;
	HIP_SAFE_CALL( hipGetDevice(&current_device) );
	HIP_SAFE_CALL( hipGetDeviceProperties(&deviceProp, current_device) );
	return deviceProp;
}

// Print basic device information
static void StoreDeviceInfo(FILE *fout){
	hipDeviceProp_t deviceProp;
	int current_device, driver_version;
	HIP_SAFE_CALL( hipGetDevice(&current_device) );
	HIP_SAFE_CALL( hipGetDeviceProperties(&deviceProp, current_device) );
	HIP_SAFE_CALL( hipDriverGetVersion(&driver_version) );
	fprintf(fout, "------------------------ Device specifications ------------------------\n");
	fprintf(fout, "Device:              %s\n", deviceProp.name);
	fprintf(fout, "CUDA driver version: %d.%d\n", driver_version/1000, driver_version%1000);
	fprintf(fout, "GPU clock rate:      %d MHz\n", deviceProp.clockRate/1000);
	//fprintf(fout, "Memory clock rate:   %d MHz\n", deviceProp.memoryClockRate/1000/2);
	//fprintf(fout, "Memory bus width:    %d bits\n", deviceProp.memoryBusWidth);
	fprintf(fout, "WarpSize:            %d\n", deviceProp.warpSize);
	fprintf(fout, "L2 cache size:       %d KB\n", deviceProp.l2CacheSize/1024);
	fprintf(fout, "Total global mem:    %d MB\n", (int)(deviceProp.totalGlobalMem/1024/1024));
	//fprintf(fout, "ECC enabled:         %s\n", deviceProp.ECCEnabled?"Yes":"No");
#ifdef __HIP_PLATFORM_NVCC__
	fprintf(fout, "Compute Capability:  %d.%d\n", deviceProp.major, deviceProp.minor);
#endif
	const int TotalSPs = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor)*deviceProp.multiProcessorCount;
	fprintf(fout, "Total SPs:           %d (%d MPs x %d SPs/MP)\n", TotalSPs, deviceProp.multiProcessorCount, _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor));
	double InstrThroughput, MemBandwidth;
	GetDevicePeakInfo(&InstrThroughput, &MemBandwidth, &deviceProp);
	fprintf(fout, "Compute throughput:  %.2f GFlops (theoretical single precision FMAs)\n", 2.0*InstrThroughput);
	fprintf(fout, "Memory bandwidth:    %.2f GB/sec\n", MemBandwidth/(1000.0*1000.0*1000.0));
	fprintf(fout, "-----------------------------------------------------------------------\n");
}

#endif
