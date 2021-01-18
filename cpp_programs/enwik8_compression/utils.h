#include <vector>
#include <fstream>
#include <cstring>

using std::vector;
using std::ifstream;
using std::strerror;
using std::runtime_error;
using std::byte;

std::vector<std::byte> load_file(std::string const& filepath);
