//
// Created by doublepmcl on 14.06.19.
//

#include "Combinations.h"

Combinations::Combinations(int m, int n) :
    _m(m),
    _n(n),
    _vals(n),
    _iter(0) {
}

Combinations::Combinations(const Combinations& obj) :
    _m(obj._m),
    _n(obj._n),
    _vals(obj._vals),
    _iter(obj._iter) {
}

Combinations::~Combinations() {
}

size_t Combinations::ownPow(size_t b, size_t e) {
  int r = 1;
  while (e)
  {
    if (e & 1)
      r *= b;
    e /= 2;
    b *= b;
  }
  return r;
}

ostream& operator<<(ostream& os, const Combinations& obj) {
  os << "iter: " << obj._iter << ", vals: [";
  for (size_t i = 0; i < obj._n; ++i) {
    if (i > 0) {
      os << ", ";
    }
    os << obj._vals[i];
  }
  os << "]";
  return os;
}

void Combinations::clearVector() {
  _iter = 0;
  _vals.resize(_n, 0);
}

// returns true only, if all values are again 0 !
bool Combinations::nextCombo() {
  bool is_all_zero = true;

  for (size_t i = 0; i < _n; ++i) {
    if ((_vals[i] = (_vals[i]+1) % _m) != 0) {
      is_all_zero = false;
      break;
    }
  }

  _iter++;

  return is_all_zero;
}

Combinations::Combinations(int m, int n) :
    _m(m),
    _n(n),
    _vals(n),
    _iter(0) {
}

Combinations::Combinations(const Combinations& obj) :
    _m(obj._m),
    _n(obj._n),
    _vals(obj._vals),
    _iter(obj._iter) {
}

Combinations::~Combinations() {
}

size_t Combinations::ownPow(size_t b, size_t e) {
  int r = 1;
  while (e)
  {
    if (e & 1)
      r *= b;
    e /= 2;
    b *= b;
  }
  return r;
}

ostream& operator<<(ostream& os, const Combinations& obj) {
  os << "iter: " << obj._iter << ", vals: [";
  for (size_t i = 0; i < obj._n; ++i) {
    if (i > 0) {
      os << ", ";
    }
    os << obj._vals[i];
  }
  os << "]";
  return os;
}

void Combinations::clearVector() {
  _iter = 0;
  _vals.resize(_n, 0);
}

// returns true only, if all values are again 0 !
bool Combinations::nextCombo() {
  bool is_all_zero = true;

  for (size_t i = 0; i < _n; ++i) {
    if ((_vals[i] = (_vals[i]+1) % _m) != 0) {
      is_all_zero = false;
      break;
    }
  }

  _iter++;

  return is_all_zero;
}

Combinations::Combinations(int m, int n) :
    _m(m),
    _n(n),
    _vals(n),
    _iter(0) {
}

Combinations::Combinations(const Combinations& obj) :
    _m(obj._m),
    _n(obj._n),
    _vals(obj._vals),
    _iter(obj._iter) {
}

Combinations::~Combinations() {
}

size_t Combinations::ownPow(size_t b, size_t e) {
  int r = 1;
  while (e)
  {
    if (e & 1)
      r *= b;
    e /= 2;
    b *= b;
  }
  return r;
}

ostream& operator<<(ostream& os, const Combinations& obj) {
  os << "iter: " << obj._iter << ", vals: [";
  for (size_t i = 0; i < obj._n; ++i) {
    if (i > 0) {
      os << ", ";
    }
    os << obj._vals[i];
  }
  os << "]";
  return os;
}

void Combinations::clearVector() {
  _iter = 0;
  _vals.resize(_n, 0);
}

// returns true only, if all values are again 0 !
bool Combinations::nextCombo() {
  bool is_all_zero = true;

  for (size_t i = 0; i < _n; ++i) {
    if ((_vals[i] = (_vals[i]+1) % _m) != 0) {
      is_all_zero = false;
      break;
    }
  }

  _iter++;

  return is_all_zero;
}

Combinations::Combinations(int m, int n) :
    _m(m),
    _n(n),
    _vals(n),
    _iter(0) {
}

Combinations::Combinations(const Combinations& obj) :
    _m(obj._m),
    _n(obj._n),
    _vals(obj._vals),
    _iter(obj._iter) {
}

Combinations::~Combinations() {
}

size_t Combinations::ownPow(size_t b, size_t e) {
  int r = 1;
  while (e)
  {
    if (e & 1)
      r *= b;
    e /= 2;
    b *= b;
  }
  return r;
}

ostream& operator<<(ostream& os, const Combinations& obj) {
  os << "iter: " << obj._iter << ", vals: [";
  for (size_t i = 0; i < obj._n; ++i) {
    if (i > 0) {
      os << ", ";
    }
    os << obj._vals[i];
  }
  os << "]";
  return os;
}

void Combinations::clearVector() {
  _iter = 0;
  _vals.resize(_n, 0);
}

// returns true only, if all values are again 0 !
bool Combinations::nextCombo() {
  bool is_all_zero = true;

  for (size_t i = 0; i < _n; ++i) {
    if ((_vals[i] = (_vals[i]+1) % _m) != 0) {
      is_all_zero = false;
      break;
    }
  }

  _iter++;

  return is_all_zero;
}
