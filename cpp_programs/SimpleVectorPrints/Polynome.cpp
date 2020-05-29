//
// Created by doublepmcl on 06.06.19.
//

#include <iostream>

#include "Polynome.h"

Polynome::Polynome(const vector<int32_t>& factors) : _factors(factors) {
  checkZerosFactors();
}

Polynome::Polynome(const Polynome& obj) : _factors(obj._factors) {
  checkZerosFactors();
}

Polynome::~Polynome() {
  _factors.clear();
}

void Polynome::shortenPolynome(const vector<int32_t>& src, vector<int32_t>& dst) {
  // While doing push_back from src to dst, first check, if a power already exists!
  size_t l1 = src.size();

  for (size_t i1 = 0; i1 < l1; ++i1) {
    const int32_t& factor1 = src.at(i1);
    bool was_found = false;
    size_t l2 = dst.size();

    for (size_t i2 = 0; i2 < l2; ++i2) {
      const int32_t& factor2 = dst.at(i2);

      // TODO: better create an vector for adding the factors up!
      if ((!was_found) & (factor2 == factor1)) {
        was_found = true;

        dst.at(i2) += factor1;
      }
    }

    if (!was_found) {
      dst.push_back(factor1);
    }
  }
}

void Polynome::shiftFactors(int32_t factor) {
  size_t l = _factors.size();

  if (factor > 0) {
    size_t new_l = l + factor;
    _factors.resize(new_l, 0);

    for (size_t i = 0; i < l; ++i) {
      _factors.at(new_l - 1 - i) = _factors.at(l - 1 - i);
      _factors.at(l - 1 - i) = 0;
    }
  } else if (factor < 0) {
    factor = -factor;

    if (l > factor) {
      size_t new_l = l - factor;

      for (size_t i = 0; i < new_l; ++i) {
        _factors.at(i) = _factors.at(i + factor);
      }

      _factors.resize(new_l);
    } else if (l <= factor) {
      _factors.clear();
    }
  }
}

void Polynome::checkZerosFactors() {
  size_t found_zeros = 0;
  size_t l = _factors.size();

  bool is_still_zero = true;
  for (size_t i = l; (i > 0) && is_still_zero; --i) {
    if (_factors.at(i - 1) != 0) {
      is_still_zero = false;
    } else {
      found_zeros += 1;
    }
  }

  if (found_zeros > 0) {
    _factors.resize(l - found_zeros, 0);
  }
}

void Polynome::multiplyWithInt(int32_t factor) {
  size_t l = _factors.size();

  for (size_t i = 0; i < l; ++i) {
    _factors.at(i) *= factor;
  }
}

const vector<int32_t>& Polynome::getFactors() const {
  return _factors;
}

ostream& operator<<(ostream& os, const Polynome& obj) {
  const vector<int32_t>& factors = obj._factors;
  size_t l = factors.size();

  if (l == 0) {
    os << "0";
  } else {
    int32_t a;
    a = factors.at(0);
    if (a < 0) {
      os << "(" << a << ")";
    } else {
      os << a;
    }

    for (size_t i = 1; i < l; ++i) {
      bool print_x = true;
      os << " + ";
      a = factors[i];
      if (a == 0) {
        print_x = false;
      } else if (a < 0) {
        os << "(" << a << ")";
      } else if (a != 1){
        os << a << "*";
      }

      if (print_x) {
        os << "x";
        if (i > 1) {
          os << "^" << i;
        }
      }
    }
  }

  return os;
}

Polynome& Polynome::operator= (Polynome const& rhs) {
  if (this != &rhs) {
    Polynome tmp(rhs);
    _factors = tmp._factors;
  }
  return *this;
}

Polynome Polynome::operator+(const Polynome& rhs) {
  vector<int32_t> f1 =  _factors;
  vector<int32_t> f2 = rhs._factors;

  size_t l1 = f1.size();
  size_t l2 = f2.size();

  size_t l = l1;

  if (l1 != l2) {
    if (l1 > l2) {
      f2.resize(l1, 0);
    } else {
      f1.resize(l2, 0);
      l = l2;
    }
  }

  for (size_t i = 0; i < l; ++i) {
    f1.at(i) += f2.at(i);
  }

  Polynome p3(f1);
  return p3;
}

Polynome Polynome::operator-(const Polynome& rhs) {
  vector<int32_t> f1 =  _factors;
  vector<int32_t> f2 = rhs._factors;

  size_t l1 = f1.size();
  size_t l2 = f2.size();

  size_t l = l1;

  if (l1 != l2) {
    if (l1 > l2) {
      f2.resize(l1, 0);
    } else {
      f1.resize(l2, 0);
      l = l2;
    }
  }

  for (size_t i = 0; i < l; ++i) {
    f1.at(i) -= f2.at(i);
  }

  Polynome p3(f1);
  return p3;
}

Polynome operator*(const Polynome& lhs, const int32_t rhs) {
  Polynome p(lhs);
  p.multiplyWithInt(rhs);
  return p;
}

Polynome operator*(const int32_t lhs, const Polynome& rhs) {
  Polynome p(rhs);
  p.multiplyWithInt(lhs);
  return p;
}

Polynome operator*(const Polynome& lhs, const Polynome& rhs) {
  Polynome p = Polynome(vector<int32_t >());

  const vector<int32_t>& factors = rhs._factors;
  size_t l = factors.size();
  for (size_t i = 0; i < l; ++i) {
    int32_t factor = factors.at(i);
    Polynome p1 = lhs * factor;
//    cout << endl << "i: " << i << ", factor: " << factor << ", p1: " << p1;
    p1.shiftFactors(i);
    p = (p + p1);
  }
//  cout << endl;

  return p;
}

bool operator==(const Polynome& lhs, const Polynome& rhs) {
  bool are_equal = true;

  const vector<int32_t> f1 = lhs._factors;
  const vector<int32_t> f2 = rhs._factors;

  size_t l1 = f1.size();
  size_t l2 = f2.size();

  if (l1 != l2) {
    are_equal = false;
  } else {
    for (size_t i = 0; (i < l1) && are_equal; ++i) {
      if (f1.at(i) != f2.at(i)) {
        are_equal = false;
      }
    }
  }

  return are_equal;
}
