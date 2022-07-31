/**
 * main-sycl.cpp: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <iostream>
#include <CL/sycl.hpp>
#include "lsyclutil.h"
#include "mix_kernels_sycl.h"
#include "version_info.h"

#define DEF_VECTOR_SIZE (32*1024*1024)

typedef struct{
    int device_index;
    bool use_os_timer;
    int wg_size;
    unsigned int vecwidth;
} ArgParams;

// Argument parsing
// returns whether program execution should continue (true) or just print help output (false)
bool argument_parsing(int argc, char* argv[], ArgParams *output){
    int arg_count = 0;
    for(int i=1; i<argc; i++) {
        if( (strcmp(argv[i], "-h")==0) || (strcmp(argv[i], "--help")==0) ) {
            return false;
        } else if( (strcmp(argv[i], "-t")==0) || (strcmp(argv[i], "--use-os-timer")==0) ) {
            output->use_os_timer = true;
        } else {
            unsigned long value = strtoul(argv[i], NULL, 10);
            switch( arg_count ){
                // device selection
                case 0:
                    output->device_index = value;
                    arg_count++;
                    break;
                // workgroup size
                case 1:
                    output->wg_size = value;
                    arg_count++;
                    break;
                // array size (x1024^2)
                case 2:
                    output->vecwidth = value;
                    arg_count++;
                    break;
                default:
                    return false;
            }
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "mixbench-sycl (" << VERSION_INFO << ")" << std::endl;

    ArgParams args{1, false, 256, DEF_VECTOR_SIZE/(1024*1024)};

    if (!argument_parsing(argc, argv, &args)) {
        std::cout << "Usage: mixbench-sycl [options] [device index [workgroup size [array size(1024^2)]]]" << std::endl
                  << std::endl
                  << "Options:" << std::endl
                  << "  -h or --help              Show this message" << std::endl
                  << "  -t or --use-os-timer      Use standard OS timer instead of SYCL profiling timer" << std::endl;

        auto devices = sycl::device::get_devices();
        std::cout << "Available SYCL devices:" << std::endl;
        int cur_dev_idx = 1;
        for(auto device:devices){
            std::cout << "  " << cur_dev_idx++ << ". " << device.get_info<sycl::info::device::name>() << '/' 
                      << device.get_platform().get_info<sycl::info::platform::name>() << std::endl;
        }
        exit(1);
    }

    std::cout << "Use \"-h\" argument to see available options" << std::endl;
    
    const size_t VEC_WIDTH = 1024*1024*args.vecwidth;
    unsigned int datasize = VEC_WIDTH*sizeof(double);

    std::unique_ptr<double[]> c(new double[VEC_WIDTH]);

    try {
        sycl::device device = sycl::device::get_devices().at(args.device_index-1);
            
        StoreDeviceInfo(device);

        const size_t totalDeviceMem = device.get_info<sycl::info::device::global_mem_size>();
        std::cout << "Total GPU memory:     " << totalDeviceMem << std::endl;
        std::cout << "Buffer size:          " << datasize/(1024*1024) << "MB" << std::endl;

        mixbenchGPU(device, c.get(), VEC_WIDTH, args.use_os_timer, args.wg_size);
    }
    catch (sycl::exception const &exc) {
        std::cerr << "SYCL exception caught: " << exc.what();
        std::exit(1);
    }

    return 0;
}
