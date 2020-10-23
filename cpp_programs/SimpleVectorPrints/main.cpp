#include <assert.h>
#include <iostream>

#include "Numbers.h"
#include "Polynome.h"
#include "Polynomial.h"

void testMultiplication() {
  cout << "Check for multiplication:" << endl;
  Polynome p1_l1 = Polynome(vector<int32_t >{-1, 1});
  Polynome p1_l2 = Polynome(vector<int32_t >{-2, 1});
  Polynome p1_r = Polynome(vector<int32_t >{2, -3, 1});

  cout << "Polynom check Nr. 1:" << endl;
  cout << " - p1_l1: " << p1_l1 << endl;
  cout << " - p1_l2: " << p1_l2 << endl;
  cout << " - p1_l1 * p1_l2: " << p1_l1*p1_l2 << endl;
  cout << " - p1_r: " << p1_r << endl;
  cout << " - p1_l1*p1_l2 == p1_r ? " << ((p1_l1*p1_l2 == p1_r) ? "Yes!" : "NOO!!!") << endl;

  Polynome p2_l1 = Polynome(vector<int32_t>{6, -5, 1});
  Polynome p2_l2 = Polynome(vector<int32_t>{-1, 1, -1, 1});
  Polynome p2_r = Polynome(vector<int32_t>{-6, 11, -12, 12, -6, 1});

  cout << "Polynom check Nr. 2:" << endl;
  cout << " - p2_l1: " << p2_l1 << endl;
  cout << " - p2_l2: " << p2_l2 << endl;
  cout << " - p2_l1 * p2_l2: " << p2_l1*p2_l2 << endl;
  cout << " - p2_r: " << p2_r << endl;
  cout << " - p2_l1*p2_l2 == p2_r ? " << ((p2_l1*p2_l2 == p2_r) ? "Yes!" : "NOO!!!") << endl;

  assert(p1_l1*p1_l2 == p1_r);
  assert(p2_l1*p2_l2 == p2_r);
}

// TODO: do div and modulo (/ %)
void testDivAndModulo() {
  cout << "Check for divide and modulo:" << endl;
}

// TODO: x^2+4*x^1+1 -> (x+1)^2
void testKroneckerAlgorithm() {
  cout << "Check for Kronecker algorithm:" << endl;
}

int main() {
  testMultiplication();
  testDivAndModulo();
  testKroneckerAlgorithm();

  Polynomial poly1 = Polynomial(vector<Polynome>{Polynome(vector<int32_t>{1, 2, -4, 5}), Polynome(vector<int32_t>{4, 2})});
  Polynomial poly2 = Polynomial(vector<Polynome>{Polynome(vector<int32_t>{3, -2}), Polynome(vector<int32_t>{6})});

  cout << "poly1: " << poly1 << endl;
  poly1.multiplyPolynomialItself();
  cout << "after multiplying: poly1: " << poly1 << endl;

  cout << "poly2: " << poly2 << endl;
  poly2.multiplyPolynomialItself();
  cout << "after multiplying poly2: " << poly2 << endl;

  return 0;
}
