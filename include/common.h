#pragma once

#include <algorithm>
#include <iostream>
#include <vector>

template <typename Op>
auto benchmark(Op op) {
  auto duration = op();  // drop first measurement
  std::vector<decltype(duration)> measurements;

  for (int i = 0; i < 32; i++) {
    duration = op();

    measurements.push_back(duration);
  }

  return *std::min_element(std::begin(measurements), std::end(measurements));
}
