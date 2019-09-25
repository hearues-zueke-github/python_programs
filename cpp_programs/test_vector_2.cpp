#include <iostream>
#include <vector>

using namespace std;

int main(int argc, char* argv[]) {
  uint64_t length = 4;

  vector<uint8_t> v;
  v.resize(length, 0);
  cout << "length: " << length << endl;

  uint32_t* p = (uint32_t*)v.data();

  p[0] = 0x12345678;

  cout << "v[0]: " << v[0]+0 << endl;
  cout << "v[1]: " << v[1]+0 << endl;
  cout << "v[2]: " << v[2]+0 << endl;
  cout << "v[3]: " << v[3]+0 << endl;

  return 0;
}
