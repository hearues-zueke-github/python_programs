//
// Created by doublepmcl on 06.06.19.
//

#include "Factors.h"

Factors::Factors(int32_t a, uint32_t b) : _a(a), _b(b) {
}

Factors::Factors(const Factors& obj) : _a(obj._a), _b(obj._b) {
}

Factors::~Factors() {
}

int32_t Factors::getA() {
  return _a;
}

uint32_t Factors::getB() {
  return _b;
}

ostream& operator<<(ostream& os, const Factors& obj) {
  int32_t a = obj._a;
  uint32_t b = obj._b;

  if (a < 0) {
    os << "(" << a << ")";
  } else {
    os << a;
  }
  if (b > 0) {
    os << "*x^" << b;
//    os << "*x^(" << b << ")";
  }

  return os;
}

Factors& Factors::operator= (Factors const& rhs)
{
  if (this != &rhs)  //oder if (*this != rhs)
  {
    /* kopiere elementweise, oder:*/
    Factors tmp(rhs); //Copy-Konstruktor
//    swap(tmp);
    _a = tmp._a;
    _b = tmp._b;
  }
  return *this; //Referenz auf das Objekt selbst zurückgeben
}

Factors Factors::operator+=(const Factors& rhs) {
  _a += rhs._a;
  return *this;
}

Factors::Factors(int32_t a, uint32_t b) : _a(a), _b(b) {
}

Factors::Factors(const Factors& obj) : _a(obj._a), _b(obj._b) {
}

Factors::~Factors() {
}

int32_t Factors::getA() {
  return _a;
}

uint32_t Factors::getB() {
  return _b;
}

ostream& operator<<(ostream& os, const Factors& obj) {
  int32_t a = obj._a;
  uint32_t b = obj._b;

  if (a < 0) {
    os << "(" << a << ")";
  } else {
    os << a;
  }
  if (b > 0) {
    os << "*x^" << b;
//    os << "*x^(" << b << ")";
  }

  return os;
}

Factors& Factors::operator= (Factors const& rhs)
{
  if (this != &rhs)  //oder if (*this != rhs)
  {
    /* kopiere elementweise, oder:*/
    Factors tmp(rhs); //Copy-Konstruktor
//    swap(tmp);
    _a = tmp._a;
    _b = tmp._b;
  }
  return *this; //Referenz auf das Objekt selbst zurückgeben
}

Factors Factors::operator+=(const Factors& rhs) {
  _a += rhs._a;
  return *this;
}
