/**
 * main.cpp: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <omp.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>

#include "mix_kernels_cpu.h"
#include "version_info.h"

constexpr auto DEF_VECTOR_SIZE_PER_THREAD = 4 * 1024 * 1024;

using ArgParams = struct { unsigned int vecwidth; };

// Argument parsing
// returns whether program execution should continue (true) or just print help
// output (false)
bool argument_parsing(int argc, char *argv[], ArgParams *output) {
  int arg_count = 0;
  for (int i = 1; i < argc; i++) {
    if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "--help") == 0)) {
      return false;
    } else {
      unsigned long value = strtoul(argv[i], NULL, 10);
      switch (arg_count) {
        // device selection
        case 0:
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

int main(int argc, char *argv[]) {
  std::cout << "mixbench-cpu (" << VERSION_INFO << ")" << std::endl;

  const auto hardware_concurrency = omp_get_max_threads();

  ArgParams args{static_cast<unsigned int>(
      hardware_concurrency * DEF_VECTOR_SIZE_PER_THREAD / (1024 * 1024))};

  if (!argument_parsing(argc, argv, &args)) {
    std::cout << "Usage: mixbench-cpu [options] [array size(1024^2)]"
              << std::endl
              << std::endl
              << "Options:" << std::endl
              << "  -h or --help              Show this message" << std::endl;

    exit(1);
  }

  std::cout << "Use \"-h\" argument to see available options" << std::endl;

  const size_t VEC_WIDTH = 1024 * 1024 * args.vecwidth;

  std::unique_ptr<double[]> c;

  c.reset(new (std::align_val_t(64)) double[VEC_WIDTH]);

  std::cout << "VEC_WIDTH: " << VEC_WIDTH << std::endl;

  std::cout << "Total threads: " << hardware_concurrency << std::endl;

  mixbenchCPU(c.get(), VEC_WIDTH);

  return 0;
}
