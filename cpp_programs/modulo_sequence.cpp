#include <iostream>
#include <string>
#include <stdlib.h>
#include <ostream>
#include <vector>

#include "utils.h"

using namespace std;

inline bool checkUnique(const vector<int>& v, vector<int>& vc, size_t m) {
  for (size_t i = 0; i < m; ++i) {
    vc[i] = 0;
  }
  for (size_t i = 0; i < m; ++i) {
    int j = v[i];
    if ((vc[j]+=1) > 1) {
      return false;
    }
  }
  return true;
}

inline void doCycle(const int a, const int c, const int m, vector<int>& v) {
  v[0] = 0;
  for (int i = 1; i < m; ++i) {
    v[i] = (a*v[i-1]+c) % m;
  }
}

void getCycles(const int m, vector<vector<int>>& M, vector<vector<int>>& factors) {
  M.clear();
  vector<int> v(m);
  vector<int> vc(m);
  for (int a = 0; a < m; ++a) {
    for (int c = 0; c < m; ++c) {
      doCycle(a, c, m, v);
      if (checkUnique(v, vc, m)) {
        M.push_back(v);
        factors.push_back(vector<int>{a, c});
      }
    }
  }
}

int getCyclesLength(const int m) {
  int l = 0;
  vector<int> v(m);
  vector<int> vc(m);
  for (int a = 0; a < m; ++a) {
    for (int c = 0; c < m; ++c) {
      doCycle(a, c, m, v);
      if (checkUnique(v, vc, m)) {
        ++l;
      }
    }
  }
  return l;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    cout << "Need m as parameter!" << endl;
    return -1;
  }
  int m = atoi(argv[1]);

  vector<vector<int>> M;
  vector<vector<int>> factors;
  getCycles(m, M, factors);
  cout << "M: " << endl << M << endl;

  cout << "factors: " << endl << factors << endl;

  int l = getCyclesLength(m);
  cout << "l: " << l << endl;

  // vector<int> lens;
  // for (int i = 1; i < m+1; ++i) {
  //   int l = getCyclesLength(i);
  //   lens.push_back(l);
  //   cout << "i: " << i << ", l: " << l << endl;
  // }
  // cout << "lens: " << lens << endl;

  return 0;
}
