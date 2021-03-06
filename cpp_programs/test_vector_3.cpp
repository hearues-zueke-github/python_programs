#include <iostream>
#include <vector>

#include "utils.h"

using namespace std;

int main(int argc, char* argv[]) {
  vector<int8_t> vec_s_1 = {1, 2, 3, -1, -2, -123};
  vector<uint8_t> vec_u_1 = {1, 2, 3, 127, 128, 129};
  vector<int16_t> vec_s_2 = {1, 2, 3, -1, -2, -123};
  vector<uint16_t> vec_u_2 = {1, 2, 3, 127, 128, 129};
  vector<int32_t> vec_s_3 = {1, 2, 3, -1, -2, -123};
  vector<uint32_t> vec_u_3 = {1, 2, 3, 127, 128, 129};
  vector<int64_t> vec_s_4 = {1, 2, 3, -1, -2, -123};
  vector<uint64_t> vec_u_4 = {1, 2, 3, 127, 128, 129};
  
  cout << "vec_s_1: " << vec_s_1 << endl;
  cout << "vec_u_1: " << vec_u_1 << endl;
  cout << endl;
  cout << "vec_s_2: " << vec_s_2 << endl;
  cout << "vec_u_2: " << vec_u_2 << endl;
  cout << endl;
  cout << "vec_u_3: " << vec_u_3 << endl;
  cout << "vec_s_3: " << vec_s_3 << endl;
  cout << endl;
  cout << "vec_s_4: " << vec_s_4 << endl;
  cout << "vec_u_4: " << vec_u_4 << endl;

  return 0;
}
