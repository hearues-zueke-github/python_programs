#include <iostream>

#include <cstdint>

#include "Matrix.h"
#include "Combinations.h"

using namespace std;

int main(int argc, char* argv[]) {
  cout << "Hello, World!" << endl;

  Matrix<int32_t> a(2, 3);
  Matrix<int32_t> b(2, 3);

  a(1, 1) = 4;
  a(1, 2) = 2;
  a(0, 1) = 5;

  b(0, 0) = -3;
  b(0, 1) = 2;
  b(1, 0) = 7;
  b(1, 1) = 7;
  b(0, 2) = 9;
  b(1, 2) = 2;

  cout << "a: " << a << endl;
  cout << "b: " << b << endl;
  cout << "a+b: " << (a+b) << endl;
  cout << "a-b: " << (a-b) << endl;
  cout << "a*b: " << (a*b) << endl;
  cout << "a/b: " << (a/b) << endl;

  Matrix<int32_t> A1(2, 2);
  Matrix<int32_t> A2(2, 2);

  A1(0, 0) = 1;
  A1(0, 1) = 2;
  A1(1, 0) = 3;
  A1(1, 1) = 4;

  A2(0, 0) = 2;
  A2(0, 1) = 6;
  A2(1, 0) = 7;
  A2(1, 1) = 8;

  cout << "A1: " << A1 << endl;
  cout << "A2: " << A2 << endl;

  Matrix<int32_t> m3 = A1.dot(A2);
  cout << "m3: " << m3 << endl;
  Matrix<int32_t> m4 = m3.mod(6);
  cout << "m4: " << m4 << endl;
  return 0;

  Combinations comb1(5, 4);
  Combinations comb2(5, 4);

  comb1.clearVector();
  cout << comb1 << endl;
  while (!comb1.nextCombo()) {
    cout << comb1 << endl;
  }

  return 0;
}