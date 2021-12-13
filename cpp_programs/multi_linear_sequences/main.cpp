#include <iostream>
#include <cmath>
#include <vector>
#include <assert.h>
#include <map>
#include <algorithm>
#include <unordered_set>

#include "utils.h"

using namespace std;

using std::cout;
using std::endl;

using std::vector;
using std::map;
using std::fill;

typedef struct GenerateAllCombVec_ {
  vector<u32> arr;
  u32 count;
  u32 n;
  u32 m;
  bool isFinished;
  GenerateAllCombVec_(const u32 n_, const u32 m_) :
    arr(n_, 0), count(0), n(n_), m(m_), isFinished(false)
  {}
  GenerateAllCombVec_(const u32 n_, const u32 m_, const u32 count_) :
    GenerateAllCombVec_(n_, m_)
  {
    u32 c = count_;
    for (u32 i = 0; i < this->n; ++i) {
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
    for (u32 i = 0; i < this->n; ++i) {
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
  inline const u64 calcIdx() {
    u64 mult = 1ull;
    u64 s = 0;
    for (u32 i = 0; i < this->n; ++i) {
      s += (u64)(this->arr[i]) * mult;
      mult *= (u64)(this->m);
    }

    return s;
  }
} GenerateAllCombVec;

typedef struct ArrPrepand_ {
  const GenerateAllCombVec* arr_a;
  u32 n;
  u32 n_2;
  u32 m;
  vector<u32> arr;
  ArrPrepand_(const GenerateAllCombVec* arr_a_) :
    arr_a(arr_a_), n(arr_a_->n), n_2(pow(arr_a_->n, 2)), m(arr_a_->m), arr(n_2, 0)
  {
    this->arr[this->n_2 - 1] = 1;
  };
  void prepand() {
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
  }
  inline const u64 calcIdx() {
    u64 mult = 1ull;
    u64 s = 0;
    for (u32 i = 0; i < this->n; ++i) {
      s += (u64)(this->arr[i]) * mult;
      mult *= (u64)(this->m);
    }

    return s;
  }
} ArrPrepand;

typedef struct VecTempIter_ {
  vector<u32> arr;
  vector<u32> arr_mult;
  u32 n;
  u32 n_2;
  u32 m;
  const ArrPrepand* arr_prep;
  const GenerateAllCombVec* arr_k;
  VecTempIter_(const ArrPrepand* arr_prep_, const GenerateAllCombVec* arr_k_) :
    arr(arr_prep_->n, 0), arr_mult(arr_prep_->n_2, 0),
    n(arr_prep_->n), n_2(arr_prep_->n_2), m(arr_prep_->m),
    arr_prep(arr_prep_), arr_k(arr_k_)
  {}
  inline void multiply() {
    for (u32 i = 0; i < this->n_2; ++i) {
      this->arr_mult[i] = (this->arr_prep->arr[i] * this->arr_k->arr[i]) % this->m;
    }
  }
  inline void shift() {
    for (u32 i = 0; i < this->n - 1; ++i) {
      this->arr[i] = this->arr_prep->arr_a->arr[i + 1];
    }
  }
  inline const u32 sum() {
    this->multiply();
    u32 s = 0;
    for (u32 i = 0; i < this->n_2; ++i) {
      s += this->arr_mult[i];
    }
    this->shift();
    return s;
  }
  inline void iterate() {
    this->arr[this->n - 1] = this->sum() % this->m;
  }
  inline const u64 calcIdxNext() {
    this->iterate();

    u64 mult = 1ull;
    u64 s = 0;
    for (u32 i = 0; i < this->n; ++i) {
      s += (u64)(this->arr[i]) * mult;
      mult *= (u64)(this->m);
    }

    return s;
  }
} VecTempIter;

// typedef struct CyclesOfKIdx_ {
//   vector<vector<u32>> vec_cycles;
//   u64 k_idx;
//   CyclesOfKIdx_(const vector<vector<u32>>& vec_cycles_, const u32 k_idx_) :
//     vec_cycles(vec_cycles_), k_idx(k_idx_)
//   {}
// } CyclesOfKIdx;

int main(int argc, char* argv[]) {
  cout << "Hello World!" << endl;

  const u32 n = 2;
  const u32 m = 5;
  GenerateAllCombVec vec_a = GenerateAllCombVec(n, m);
  GenerateAllCombVec vec_k = GenerateAllCombVec(pow(n, 2), m);

  ArrPrepand arr_prep = ArrPrepand(&vec_a);

  VecTempIter vec_temp_iter = VecTempIter(&arr_prep, &vec_k);

  vector<u32> arr_idx_to_idx_next(pow(m, n));

  map<u64, map<u64, u64>> map_k_idx_to_map_a_idx_to_idx_next;
  map<u64, vector<vector<u64>>> map_k_idx_to_vec_cycles;
  // vector<vector<u32>> vec_all_cycles;
  map<u32, u32> map_len_cycle_to_count;

  while (!vec_k.isFinished) { // && vec_k.count < 1000) {
    const u64 k_idx = vec_k.calcIdx();
    vec_a.reset();
    while (!vec_a.isFinished) { // && vec_a.count < 300) {
      arr_prep.prepand();

      const u64 idx_now = vec_a.calcIdx();
      const u64 idx_next = vec_temp_iter.calcIdxNext();

      arr_idx_to_idx_next[idx_now] = idx_next;

      vec_a.next();
    }

    unordered_set<u64> set_idx_used;
    vector<vector<u64>> vec_cycles;

    unordered_set<u64> set_one_cycle;
    vector<u64> vec_one_cycle;
    map<u64, u64> map_a_idx_to_idx_next;
    for (u64 i = 0; i < arr_idx_to_idx_next.size(); ++i) {
      map_a_idx_to_idx_next[i] = arr_idx_to_idx_next[i];
    }

    map_k_idx_to_map_a_idx_to_idx_next.emplace(k_idx, map_a_idx_to_idx_next);

    cout << "k_idx: " << k_idx << endl;
    cout << "map_a_idx_to_idx_next: " << map_a_idx_to_idx_next << endl;

    //if (vec_k.count > 231) {
    //  exit(0);
    //}

    while (map_a_idx_to_idx_next.size() > 0) {
      set_one_cycle.clear();
      vec_one_cycle.clear();

      const auto t_1 = map_a_idx_to_idx_next.begin();
      const u64 idx_now_1 = t_1->first;
      const u64 idx_next_1 = t_1->second;
      map_a_idx_to_idx_next.erase(t_1);

      if (idx_now_1 == idx_next_1) {
          vec_one_cycle.emplace_back(idx_now_1);
          vec_cycles.emplace_back(vec_one_cycle);

          set_idx_used.insert(idx_now_1);
          continue;
      }

      if (set_idx_used.find(idx_next_1) != set_idx_used.end()) {
        continue;
      }

      set_one_cycle.insert(idx_now_1);
      vec_one_cycle.push_back(idx_now_1);

      set_one_cycle.insert(idx_next_1);
      vec_one_cycle.push_back(idx_next_1);

      bool is_not_cycle = true;
      u64 idx_now = idx_next_1;
      while (true) {
        const u64 idx_next = map_a_idx_to_idx_next[idx_now];
        map_a_idx_to_idx_next.erase(idx_now);

        if (set_idx_used.find(idx_next) != set_idx_used.end()){
          break;
        }

        if (set_one_cycle.find(idx_next) != set_one_cycle.end()){
          const auto iter = std::find(vec_one_cycle.begin(), vec_one_cycle.end(), idx_next);

          vector<u64> vec_one_cycle_true;
          for (auto it = iter; it != vec_one_cycle.end(); ++it) {
            const u32 idx = *it;
            vec_one_cycle_true.push_back(idx);
            set_idx_used.insert(idx);
          }

          vec_cycles.emplace_back(vec_one_cycle_true);
          // find the cycle! and extract it
          break;
        }

        set_one_cycle.insert(idx_next);
        vec_one_cycle.push_back(idx_next);

        idx_now = idx_next;
      }
    }

    // if ((vec_k.count % 10000) == 0) {
      cout << "vec_k: " << vec_k.arr << endl;
      cout << "vec_cycles: " << vec_cycles << endl;
    // }
    // break;

    // vec_cylces_of_k.emplace_back(vec_cycles, vec_k.calcIdx());

    for (const vector<u64>& vec_cycle : vec_cycles) {
      const u32 len = (u32)vec_cycle.size();
      if (map_len_cycle_to_count.find(len) == map_len_cycle_to_count.end()) {
        map_len_cycle_to_count.emplace(len, 0);
      }
      map_len_cycle_to_count[len] += 1;
    }

    map_k_idx_to_vec_cycles.emplace(k_idx, vec_cycles);
    // cout << "- arr_idx_to_idx_next: " << arr_idx_to_idx_next << endl;
    vec_k.next();
  }

  // cout << "map_k_idx_to_vec_cycles: " << map_k_idx_to_vec_cycles << endl;
  cout << "map_len_cycle_to_count: " << map_len_cycle_to_count << endl;
  // cout << "map_k_idx_to_map_a_idx_to_idx_next: " << map_k_idx_to_map_a_idx_to_idx_next << endl;

  return 0;
}
