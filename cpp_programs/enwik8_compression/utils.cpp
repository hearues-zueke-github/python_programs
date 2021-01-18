#include "utils.h"

vector<byte> load_file(std::string const& filepath)
{
  std::ifstream ifs(filepath, std::ios::binary|std::ios::ate);

  if(!ifs) {
    throw runtime_error(filepath + ": " + std::strerror(errno));
  }

  auto end = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  auto size = std::size_t(end - ifs.tellg());

  if(size == 0) { // avoid undefined behavior
    return {};
  }

  vector<byte> buffer(size);

  if(!ifs.read((char*)buffer.data(), buffer.size()))
    throw runtime_error(filepath + ": " + std::strerror(errno));

  return buffer;
}
