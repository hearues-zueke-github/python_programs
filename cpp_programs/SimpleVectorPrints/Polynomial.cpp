//
// Created by doublepmcl on 06.06.19.
//

#include "Polynomial.h"

Polynomial::Polynomial(const vector<Polynome>& polynomes) : _polynomes(polynomes) {
}

Polynomial::Polynomial(const Polynomial& obj) : _polynomes(obj._polynomes) {
}

Polynomial::~Polynomial() {
  _polynomes.clear();
}

void Polynomial::multiplyPolynomialItself() {
  Polynome p = Polynome(vector<int32_t>{1});
//  cout << "start p: " << p << endl;

  size_t l = _polynomes.size();
  for (size_t i = 0; i < l; ++i) {
    p = p * _polynomes.at(i);
//    cout << "i: " << i << ", p: " << p << endl;
  }

  _polynomes.clear();
  _polynomes.push_back(p);
}

ostream& operator<<(ostream& os, const Polynomial& obj) {
  const vector<Polynome>& polynomes = obj._polynomes;
  size_t l = polynomes.size();

  if (l == 0) {
    os << "0";
  } else if (l == 1) {
    os << polynomes.at(0);
  } else {
    for (size_t i = 0; i < l; ++i) {
      if (i > 0) {
        os << " * ";
      }
      os << "(" << polynomes.at(i) << ")";
    }
  }

  return os;
}
