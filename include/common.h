#pragma once

#include <algorithm>
#include <iostream>
#include <vector>

template <typename Op>
auto benchmark(Op op) {
  constexpr int total_runs = 20;
  auto duration = op();  // drop first measurement
  std::vector<decltype(duration)> measurements;
  for (int i = 1; i < total_runs; i++) {
    duration = op();
    measurements.push_back(duration);
  }
  return *std::min_element(std::begin(measurements), std::end(measurements));
}
