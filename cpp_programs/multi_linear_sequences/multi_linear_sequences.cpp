#include "multi_linear_sequences.h"

// k_idx_start: inclusive, k_idx_end: exclusive
void calcCycleLengthAmounts(InputTypeOwn& inp_var, ReturnTypeOwn& ret_var) {
  const U32 n = inp_var.n;
  const U32 m = inp_var.m;
  const U64 k_idx_start = inp_var.k_idx_start;
  const U64 k_idx_end = inp_var.k_idx_end;
  std::map<U32, U32>& map_len_cycle_to_count = ret_var.map_len_cycle_to_count;

  GenerateAllCombVec vec_a = GenerateAllCombVec(n, m);
  GenerateAllCombVec vec_k = GenerateAllCombVec(pow(n, 2), m, k_idx_start);

  ArrPrepand arr_prep = ArrPrepand(&vec_a);

  VecTempIter vec_temp_iter = VecTempIter(&arr_prep, &vec_k);

  std::vector<U32> arr_idx_to_idx_next(pow(m, n));

  std::map<U64, std::vector<std::vector<U64>>> map_k_idx_to_vec_cycles;

  while (!vec_k.isFinished) {
    const U64 k_idx = vec_k.calcIdx();
    if (k_idx >= k_idx_end) {
      break;
    }

    vec_a.reset();
    while (!vec_a.isFinished) {
      arr_prep.prepand();

      const U64 idx_now = vec_a.calcIdx();
      const U64 idx_next = vec_temp_iter.calcIdxNext();

      arr_idx_to_idx_next[idx_now] = idx_next;

      vec_a.next();
    }

    unordered_set<U64> set_idx_used;
    std::vector<std::vector<U64>> vec_cycles;

    unordered_set<U64> set_one_cycle;
    std::vector<U64> vec_one_cycle;
    std::map<U64, U64> map_a_idx_to_idx_next;
    for (U64 i = 0; i < arr_idx_to_idx_next.size(); ++i) {
      map_a_idx_to_idx_next[i] = arr_idx_to_idx_next[i];
    }

    while (map_a_idx_to_idx_next.size() > 0) {
      set_one_cycle.clear();
      vec_one_cycle.clear();

      const auto t_1 = map_a_idx_to_idx_next.begin();
      const U64 idx_now_1 = t_1->first;
      const U64 idx_next_1 = t_1->second;

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

      U64 idx_now = idx_next_1;
      while (true) {
        if (!map_a_idx_to_idx_next.contains(idx_now)) {
          break;
        }

        const U64 idx_next = map_a_idx_to_idx_next[idx_now];
        map_a_idx_to_idx_next.erase(idx_now);

        if (set_idx_used.find(idx_next) != set_idx_used.end()) {
          break;
        }

        if (set_one_cycle.find(idx_next) != set_one_cycle.end()) {
          const auto iter = std::find(vec_one_cycle.begin(), vec_one_cycle.end(), idx_next);

          std::vector<U64> vec_one_cycle_true;
          for (auto it = iter; it != vec_one_cycle.end(); ++it) {
            const U32 idx = *it;
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

    for (const std::vector<U64>& vec_cycle : vec_cycles) {
      const U32 len = (U32)vec_cycle.size();
      if (map_len_cycle_to_count.find(len) == map_len_cycle_to_count.end()) {
        map_len_cycle_to_count.emplace(len, 0);
      }
      map_len_cycle_to_count[len] += 1;
    }

    map_k_idx_to_vec_cycles.emplace(k_idx, vec_cycles);
    vec_k.next();
  }
}
