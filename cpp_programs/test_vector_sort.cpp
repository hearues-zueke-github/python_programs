#include <iostream>
#include <algorithm>
#include <chrono>
#include <random>
#include <vector>
#include <tuple>

#include "utils.h"

int main(int argc, char* argv[]) {
  mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());  
  auto uni_dist = std::uniform_int_distribution<int>(0, 100);
  
  std::vector<std::tuple<int, int>> v;
  // std::vector<int> v;
  for (int i = 0; i < 10; ++i) {
    v.push_back(std::make_tuple(uni_dist(rng), uni_dist(rng)));
  }

  cout << "v: " << v << endl;

  return 0;
}
