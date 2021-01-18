#include <iostream>
#include <map>

#include "utils.h"

using namespace std;

using i32 = int;
using u32 = unsigned int;

using std::cout;
using std::endl;
using std::map;

int main(int argc, char* argv[]) {
  cout << "Hello, World!" << endl;

  vector<byte> content = load_file("/home/doublepmcl/Downloads/enwik8");

  content.resize(10000000);

  // map<vector<byte>, u32> cont_cnt;
  
  map<byte, map<byte, map<byte, vector<u32>>>> cont_pos;
  // map<vector<byte>, vector<u32>> cont_pos;

  cout << "content.size(): " << content.size() << endl;

  const u32 length = content.size();

  u32 chunck_size = 3;

  const u32 length_loop = length - chunck_size + 1;
  vector<byte> v(chunck_size, (byte)0);
  // auto copy_vector = [&v, &chunck_size](const vector<byte>::iterator& iter) {
  //   for (u32 i = 0; i < chunck_size; ++i) {
  //     v[i] = *(iter + i);
  //   }
  // };

  for (u32 i = 0 ; i < length_loop; ++i) {
    if (i % 10000 == 0) {
      cout << "i: " << i << endl;
    }

    // auto i1 = content.begin() + i;
    // std::copy(i1, i1 + chunck_size, v.begin());

    // // copy_vector(content.begin() + i);

    // if (cont_pos.find(v) == cont_pos.end()) {
    //   cont_pos.insert({v, {}});
    // }

    // cont_pos[v].push_back(i);

    const byte& b1 = content[i];
    const byte& b2 = content[i + 1];
    const byte& b3 = content[i + 2];

    if (cont_pos.find(b1) == cont_pos.end()) {
      cont_pos.insert({b1, {}});
    }
    auto& cont_pos_1 = cont_pos[b1];
    
    if (cont_pos_1.find(b2) == cont_pos_1.end()) {
      cont_pos_1.insert({b2, {}});
    }
    auto& cont_pos_2 = cont_pos_1[b2];

    if (cont_pos_2.find(b3) == cont_pos_2.end()) {
      cont_pos_2.insert({b3, {}});
    }
    auto& cont_pos_3 = cont_pos_2[b3];

    cont_pos_3.push_back(i);
  }

  cout << "chunck_size: " << chunck_size << endl;
  cout << "content.size(): " << content.size() << endl;

  return 0;
}
