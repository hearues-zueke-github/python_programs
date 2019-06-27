#include <iostream>
#include <vector>

#include "utils.h"

using namespace std;

int main(int argc, char* argv[]) {
  if (argc < 2) {
    cout << "Need length as parameter!" << endl;
    return -1;
  }
  uint64_t length = (uint64_t)atoi(argv[1]);

  vector<uint64_t> v;
  v.resize(length, 0);
  cout << "length: " << length << endl;

  for (uint64_t i = 0; i < length; ++i) {
    cout << "i: " << i << endl;
    v[i] = i;
  }

  vector<uint64_t> vr;
  for (uint64_t i = 0; i < length; ++i) {
    // vr.push_back(v[v.size()-1-i]);
    vr.insert(vr.begin(), v[i]);
  }

  cout << "vr: " << vr << endl;

  // vector<uint64_t>* v = new vector<uint64_t>();
  // v->resize(length, 0);
  // cout << "length: " << length << endl;

  // for (uint64_t i = 0; i < length; ++i) {
  //   cout << "i: " << i << endl;
  //   (*v)[i];
  // }

  return 0;
}
