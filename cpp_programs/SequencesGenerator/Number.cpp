//
// Created by doublepmcl on 14.06.19.
//

#include "Number.h"

template<typename T>
Number<T>::Number(T n) : _n(n) {
}

template<typename T>
Number<T>::Number(const Number<T>& obj) : _n(obj._n) {
}

template<typename T>
Number<T>::~Number() {
}

template<typename T>
Number<T> operator+(const Number<T>& lhs, const Number<T>& rhs) {
  return Number<T>(lhs._n + rhs._n);
}

template<typename T>
Number<T> operator%(const Number<T>& lhs, const Number<T>& rhs) {
  return Number<T>(lhs._n % rhs._n);
}

template class Number<int32_t>;
template class Number<float>;
template class Number<double>;
template class Number<long double>;

template Number<int32_t> operator+(const Number<int32_t>& lhs, const Number<int32_t>& rhs);
template Number<float> operator+(const Number<float>& lhs, const Number<float>& rhs);
template Number<double> operator+(const Number<double>& lhs, const Number<double>& rhs);
template Number<long double> operator+(const Number<long double>& lhs, const Number<long double>& rhs);

template Number<int32_t> operator%(const Number<int32_t>& lhs, const Number<int32_t>& rhs);
Number<float> operator%(const Number<float>& lhs, const Number<float>& rhs) {
  return Number<float>(fmodf(lhs._n, rhs._n));
}
Number<double> operator%(const Number<double>& lhs, const Number<double>& rhs) {
  return Number<double>(fmod(lhs._n, rhs._n));
}
Number<long double> operator%(const Number<long double>& lhs, const Number<long double>& rhs) {
  return Number<long double>(fmod(lhs._n, rhs._n));
}
