#pragma once

#include <cmath>
#include <vector>
#include <assert.h>

#include "utils.h"
#include "own_types.h"
#include "thread_data_generic.h"

struct GenerateAllCombVec_;
struct ArrPrepand_;
struct VecTempIter_;

typedef struct GenerateAllCombVec_ GenerateAllCombVec;
typedef struct ArrPrepand_ ArrPrepand;
typedef struct VecTempIter_ VecTempIter;

struct GenerateAllCombVec_ {
  vector<U32> arr;
  U32 count;
  U32 dim;
  U32 n;
  U32 m;
  bool isFinished;
  GenerateAllCombVec_(const U32 dim_, const U32 n_, const U32 m_) :
    arr(n_, 0), count(0), dim(dim_), n(n_), m(m_), isFinished(false)
  {}
  GenerateAllCombVec_(const U32 dim_, const U32 n_, const U32 m_, const U32 count_) :
    GenerateAllCombVec_(dim_, n_, m_)
  {
    U32 c = count_;
    for (U32 i = 0; i < this->n; ++i) {
      this->arr[i] = c % this->m;
      c /= this->m;
    }
    if (c > 0) {
      this->isFinished = true;
    }
  }
  const bool next() {
    if (isFinished) {
      return true;
    }

    count += 1;
    for (U32 i = 0; i < this->n; ++i) {
      if ((this->arr[i] += 1) < this->m) {
        return false;
      }
      this->arr[i] = 0;
    }

    isFinished = true;
    return true;
  }
  void reset() {
    count = 0;
    isFinished = false;
    fill(this->arr.begin(), this->arr.end(), 0);
  }
  inline const U64 calcIdx() {
    U64 mult = 1ull;
    U64 s = 0;
    for (U32 i = 0; i < this->n; ++i) {
      s += (U64)(this->arr[i]) * mult;
      mult *= (U64)(this->m);
    }

    return s;
  }
};

struct ArrPrepand_ {
  const GenerateAllCombVec* arr_a;
  U32 dim;
  U32 n;
  U32 n_2;
  U32 m;
  vector<U32> arr;
  ArrPrepand_(const GenerateAllCombVec* arr_a_) :
    arr_a(arr_a_), dim(arr_a_->dim), n(arr_a_->n), n_2(pow(dim + 1, n)), m(arr_a_->m), arr(n_2, 0)
  {
    this->arr[this->n_2 - 1] = 1;
  };
  inline void prepand() {
    switch (this->dim) {
      case 1:
        switch (this->n) {
          case 1:
            this->arr[0] = this->arr_a->arr[0];
            break;
          case 2:
            this->arr[0] = (this->arr_a->arr[0] * this->arr_a->arr[1]) % this->m;
            this->arr[1] = this->arr_a->arr[0];
            this->arr[2] = this->arr_a->arr[1];
            break;
          case 3:
            this->arr[0] = (this->arr_a->arr[0] * this->arr_a->arr[1] * this->arr_a->arr[2]) % this->m;
            this->arr[1] = (this->arr_a->arr[0] * this->arr_a->arr[1]) % this->m;
            this->arr[2] = (this->arr_a->arr[0] * this->arr_a->arr[2]) % this->m;
            this->arr[3] = (this->arr_a->arr[1] * this->arr_a->arr[2]) % this->m;
            this->arr[4] = this->arr_a->arr[0];
            this->arr[5] = this->arr_a->arr[1];
            this->arr[6] = this->arr_a->arr[2];
            break;
          default:
            assert(false && "Not implemented for n > 3!");
            break;
        }
        break;
      case 2:
        switch (this->n) {
          case 1:
            this->arr[0] = this->arr_a->arr[0];
            this->arr[1] = (this->arr_a->arr[0]*this->arr_a->arr[0]) % this->m;
            break;
          case 2:
            this->arr[0] = (this->arr_a->arr[0]*this->arr_a->arr[0] * this->arr_a->arr[1]*this->arr_a->arr[1]) % this->m;
            this->arr[1] = (this->arr_a->arr[0]*this->arr_a->arr[0] * this->arr_a->arr[1]) % this->m;
            this->arr[2] = (this->arr_a->arr[0] * this->arr_a->arr[1]*this->arr_a->arr[1]) % this->m;
            this->arr[3] = (this->arr_a->arr[0]*this->arr_a->arr[0]) % this->m;
            this->arr[4] = (this->arr_a->arr[1]*this->arr_a->arr[1]) % this->m;
            this->arr[5] = (this->arr_a->arr[0] * this->arr_a->arr[1]) % this->m;
            this->arr[6] = this->arr_a->arr[0];
            this->arr[7] = this->arr_a->arr[1];
            break;
          default:
            assert(false && "Not implemented for n > 1!");
            break;
        }
        break;
      default:
        assert(false && "Not implemented for dim > 1!");
        break;
    }
  }
  inline const U64 calcIdx() {
    U64 mult = 1ull;
    U64 s = 0;
    for (U32 i = 0; i < this->n; ++i) {
      s += (U64)(this->arr[i]) * mult;
      mult *= (U64)(this->m);
    }

    return s;
  }
};

struct VecTempIter_ {
  vector<U32> arr;
  vector<U32> arr_mult;
  U32 n;
  U32 n_2;
  U32 m;
  const ArrPrepand* arr_prep;
  const GenerateAllCombVec* arr_k;
  VecTempIter_(const ArrPrepand* arr_prep_, const GenerateAllCombVec* arr_k_) :
    arr(arr_prep_->n, 0), arr_mult(arr_prep_->n_2, 0),
    n(arr_prep_->n), n_2(arr_prep_->n_2), m(arr_prep_->m),
    arr_prep(arr_prep_), arr_k(arr_k_)
  {}
  inline void multiply() {
    for (U32 i = 0; i < this->n_2; ++i) {
      this->arr_mult[i] = (this->arr_prep->arr[i] * this->arr_k->arr[i]) % this->m;
    }
  }
  inline void shift() {
    for (U32 i = 0; i < this->n - 1; ++i) {
      this->arr[i] = this->arr_prep->arr_a->arr[i + 1];
    }
  }
  inline const U32 sum() {
    this->multiply();
    U32 s = 0;
    for (U32 i = 0; i < this->n_2; ++i) {
      s += this->arr_mult[i];
    }
    this->shift();
    return s;
  }
  inline void iterate() {
    this->arr[this->n - 1] = this->sum() % this->m;
  }
  inline const U64 calcIdxNext() {
    this->iterate();

    U64 mult = 1ull;
    U64 s = 0;
    for (U32 i = 0; i < this->n; ++i) {
      s += (U64)(this->arr[i]) * mult;
      mult *= (U64)(this->m);
    }

    return s;
  }
};

typedef struct ThreadDataGeneric<InputTypeOwn, ReturnTypeOwn> ThreadData;

void calcCycleLengthAmounts(InputTypeOwn& inp_var, ReturnTypeOwn& ret_var);
