#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "fmt/core.h"
#include "fmt/format.h"
#include "fmt/ranges.h"

using std::array;
using std::copy;
using std::cout;
using std::endl;
using std::fstream;
using std::function;
using std::getline;
using std::iota;
using std::map;
using std::ostream;
using std::set;
using std::stoi;
using std::string;
using std::to_string;
using std::vector;

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::nanoseconds;
using std::chrono::time_point_cast;

namespace fs = std::filesystem;
using fs::path;

using fmt::format;
