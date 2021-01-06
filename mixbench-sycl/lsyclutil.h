/**
 * lsyclutil.h: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#ifndef _CUTIL_H_
#define _CUTIL_H_

#include <CL/sycl.hpp>
#include <stdio.h>

using namespace cl;

#define FRACTION_CEILING(numerator, denominator) ((numerator+denominator-1)/(denominator))


// Print basic device information
inline void StoreDeviceInfo(const sycl::device &device){
    auto platform = device.get_platform();
    try{
        auto platform_name = platform.get_info<sycl::info::platform::name>();
        auto device_name = device.get_info<sycl::info::device::name>();
        auto vendor_name = device.get_info<sycl::info::device::vendor>();
        auto device_drv = device.get_info<sycl::info::device::driver_version>();

        auto device_addrbits = device.get_info<sycl::info::device::address_bits>();
        auto device_freq = device.get_info<sycl::info::device::max_clock_frequency>();
        auto device_gmem = device.get_info<sycl::info::device::global_mem_size>();
        auto device_maxalloc = device.get_info<sycl::info::device::max_mem_alloc_size>();
        auto device_syclver = device.get_info<sycl::info::device::version>();
        auto device_CUs = device.get_info<sycl::info::device::max_compute_units>();

        std::cout << "------------------------ Device specifications ------------------------" << std::endl;
        std::cout << "Platform:            " << platform_name << std::endl;
        std::cout << "Device:              " << device_name << '/' << vendor_name << std::endl;
        std::cout << "Driver version:      " << device_drv << std::endl;
        std::cout << "Address bits:        " << device_addrbits << std::endl;
        std::cout << "GPU clock rate:      " << device_freq << " MHz" << std::endl;
        std::cout << "Total global mem:    " << device_gmem/1024/1024 << " MB" << std::endl;
        std::cout << "Max allowed buffer:  " << device_maxalloc/1024/1024 << " MB" << std::endl;
        std::cout << "SYCL version:        " << device_syclver << std::endl;
        std::cout << "Total CUs:           " << device_CUs << std::endl;
        std::cout << "-----------------------------------------------------------------------" << std::endl;
    }
    catch (sycl::exception const &exc) {
        std::cerr << "Could not get full device info: ";
        std::cerr << exc.what() << std::endl;
    }
}

inline size_t GetMaxDeviceWGSize(const sycl::device &device){
    auto wgsize = device.get_info<sycl::info::device::max_work_group_size>();
    return wgsize;
}


#endif
