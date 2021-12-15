#include <iostream>
#include <cmath>
#include <vector>
#include <assert.h>
#include <map>
#include <algorithm>
#include <thread>
#include <string>
#include <queue>
#include <condition_variable>
#include <unordered_set>

#include "utils.h"

using namespace std;

using std::cout;
using std::endl;

using std::vector;
using std::map;
using std::fill;

// // add later to the header file!
// typedef struct InputType_ InputType;
// typedef struct ReturnType_ ReturnType;

typedef struct GenerateAllCombVec_ {
  vector<U32> arr;
  U32 count;
  U32 n;
  U32 m;
  bool isFinished;
  GenerateAllCombVec_(const U32 n_, const U32 m_) :
    arr(n_, 0), count(0), n(n_), m(m_), isFinished(false)
  {}
  GenerateAllCombVec_(const U32 n_, const U32 m_, const U32 count_) :
    GenerateAllCombVec_(n_, m_)
  {
    this->setCount(count_);
    this->count = count_;
  }
  inline void setCount(U32 count) {
    for (U32 i = 0; i < this->n; ++i) {
      this->arr[i] = count % this->m;
      count /= this->m;
    }
    if (count > 0) {
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
} GenerateAllCombVec;

typedef struct ArrPrepand_ {
  const GenerateAllCombVec* arr_a;
  U32 n;
  U32 n_2;
  U32 m;
  vector<U32> arr;
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
  inline const U64 calcIdx() {
    U64 mult = 1ull;
    U64 s = 0;
    for (U32 i = 0; i < this->n; ++i) {
      s += (U64)(this->arr[i]) * mult;
      mult *= (U64)(this->m);
    }

    return s;
  }
} ArrPrepand;

typedef struct VecTempIter_ {
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
} VecTempIter;

// typedef struct CyclesOfKIdx_ {
//   vector<vector<U32>> vec_cycles;
//   U64 k_idx;
//   CyclesOfKIdx_(const vector<vector<U32>>& vec_cycles_, const U32 k_idx_) :
//     vec_cycles(vec_cycles_), k_idx(k_idx_)
//   {}
// } CyclesOfKIdx;

typedef enum ThreadState_ {
  NONE = 0,
  START,
  END,
} ThreadState;

typedef struct InputTypeOwn_ InputTypeOwn;
typedef struct ReturnTypeOwn_ ReturnTypeOwn;

template<typename InputType, typename ReturnType>
struct ThreadDataGeneric;

typedef struct ThreadDataGeneric<InputTypeOwn, ReturnTypeOwn> ThreadData;

template<typename InputType, typename ReturnType>
struct ThreadDataGeneric {
  std::mutex mutex;
  std::condition_variable cond_var;
  bool main_notify_thread;
  bool thread_notify_main;
  ThreadState state;
  // bool kill_thread;
  // bool is_killed;
  InputType var_input;
  ReturnType var_return;
  
  // void (*function)(InputType& var_input, ReturnType& var_return);

  // ThreadDataGeneric(
  //   std::mutex mutex_,
  //   std::condition_variable cond_var_,
  //   void (*function_)(InputType& var_input, ReturnType& var_return),
  //   InputType var_input_,
  //   ReturnType var_return_
  // ) : mutex(mutex_), cond_var(cond_var_),
  //     main_notify_thread(false), thread_notify_main(false),
  //     // kill_thread(false), is_killed(false),
  //     state(NONE),
  //     var_input(var_input_), var_return(var_return_),
  //     function(function_) {
  // }
  ThreadDataGeneric(
    // void* function_
    // void (*function_)(InputType& var_input, ReturnType& var_return)
  ) : mutex(), cond_var(),
      main_notify_thread(false), thread_notify_main(false),
      // kill_thread(false), is_killed(false),
      state(NONE),
      var_input(), var_return() {
      // function((void (*)(InputType&, ReturnType&))function_) {
      // function(function_) {
  }
  // ThreadDataGeneric(const ThreadDataGeneric&) = delete;
  // ThreadDataGeneric(ThreadDataGeneric&&) = delete;
};

void calcCycleLengthAmounts(InputTypeOwn& inp_var, ReturnTypeOwn& ret_var);

// template<typename InputType, typename ReturnType>
// void start(struct ThreadDataGeneric<InputType, ReturnType>* const self) {
template<typename T>
void start(T self) {
// void start(struct ThreadDataGeneric<InputType, ReturnType>* const self) {
// void start(struct ThreadDataGeneric<InputTypeOwn, ReturnTypeOwn>* const self) {
  while (true) {
    { // wait in thread on start of main thread
      std::unique_lock<std::mutex> lk(self->mutex);
      self->cond_var.wait(lk, [&]{ return self->main_notify_thread; });
    }

    bool is_end = false;  
    switch (self->state) {
      case ThreadState::NONE:
        assert(false && "ThreadState state should not be NONE!");
        break;
      case ThreadState::START:
        // execute the function in the thread!
        calcCycleLengthAmounts(self->var_input, self->var_return);
        // (*self->function)(self->var_input, self->var_return);
        break;
      case ThreadState::END:
        is_end = true;
        break;
    }

    { // notify the main thread that self thread worker is finished
      std::lock_guard<std::mutex> lk(self->mutex);
      self->thread_notify_main = true;
      self->main_notify_thread = false;
      self->state = ThreadState::NONE;
    }
    self->cond_var.notify_one();

    if (is_end) {
      return;
    }
  }
}
template void start(ThreadData* const self);
// template void start(struct ThreadDataGeneric<InputTypeOwn, ReturnTypeOwn>* const self);

typedef struct InputTypeOwn_ {
  U32 n;
  U32 m;
  U64 k_idx_start;
  U64 k_idx_end;
} InputTypeOwn;

typedef struct ReturnTypeOwn_ {
  map<U32, U32> map_len_cycle_to_count;
} ReturnTypeOwn;

// typedef struct ThreadDataGeneric<InputType, ReturnType> ThreadData;

// k_idx_start: inclusive, k_idx_end: exclusive
void calcCycleLengthAmounts(InputTypeOwn& inp_var, ReturnTypeOwn& ret_var) {
// void calcCycleLengthAmounts(const U32 n, const U32 m, const U64 k_idx_start, const U64 k_idx_end, map<U32, U32>& map_len_cycle_to_count) {
  
  const U32 n = inp_var.n;
  const U32 m = inp_var.m;
  const U64 k_idx_start = inp_var.k_idx_start;
  const U64 k_idx_end = inp_var.k_idx_end;
  map<U32, U32>& map_len_cycle_to_count = ret_var.map_len_cycle_to_count;

  // cout << "n: " << n << ", m: " << m << endl;
  GenerateAllCombVec vec_a = GenerateAllCombVec(n, m);
  GenerateAllCombVec vec_k = GenerateAllCombVec(pow(n, 2), m, k_idx_start);

  ArrPrepand arr_prep = ArrPrepand(&vec_a);

  VecTempIter vec_temp_iter = VecTempIter(&arr_prep, &vec_k);

  vector<U32> arr_idx_to_idx_next(pow(m, n));

  // map<U64, map<U64, U64>> map_k_idx_to_map_a_idx_to_idx_next;
  map<U64, vector<vector<U64>>> map_k_idx_to_vec_cycles;
  // vector<vector<U32>> vec_all_cycles;

  // bool is_printing = (k_idx == 125 ? true : false);

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
    vector<vector<U64>> vec_cycles;

    unordered_set<U64> set_one_cycle;
    vector<U64> vec_one_cycle;
    map<U64, U64> map_a_idx_to_idx_next;
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

          vector<U64> vec_one_cycle_true;
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

    for (const vector<U64>& vec_cycle : vec_cycles) {
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

// TODO: make a function for aggregating the return values from the threads!
void calcCycleLengthAmountsMultiThreadingMainThread(const U32 n, const U32 m_min, const U32 m_max, const U32 cpu_amount, map<U32, map<U32, U32>>& map_m_to_map_len_cycle_to_count) {
  // vector<std::mutex> vec_mutex(cpu_amount);
  // vector<std::condition_variable> vec_cond_var(cpu_amount);
  // vector<map<U32, U32>> vec_map_len_cycle_to_count(cpu_amount);

  // vector<InputTypeOwn> vec_inp_var(cpu_amount);
  // vector<ReturnTypeOwn> vec_ret_var(cpu_amount);
  // return;

  vector<ThreadData> vec_thread_data(cpu_amount);
  // for (U32 i = 0; i < cpu_amount; ++i) {
  //   vec_thread_data.emplace_back(
  //     // (void*)&calcCycleLengthAmounts
  //     // vec_mutex[i],
  //     // vec_cond_var[i],
  //     // vec_inp_var[i],
  //     // vec_ret_var[i]
  //   );
  // }

  // InputTypeOwn inp_var;
  // ReturnTypeOwn ret_var;

  vector<std::thread> threads;
  for (U32 i = 0; i < cpu_amount; ++i) {
    ThreadData& thread_data = vec_thread_data[i];
    std::thread th([&thread_data](){start(&thread_data);});
    threads.push_back(std::move(th));
  }

  map<U32, U32> map_len_cycle_to_count_acc;
  for (U32 m = m_min; m <= m_max; ++m) {
    cout << "n: " << n << ", m: " << m << endl;

    map_len_cycle_to_count_acc.clear();
    // ret_var.map_len_cycle_to_count.clear();

    typedef struct KIdxStartEnd_ {
      U64 k_idx_start;
      U64 k_idx_end;
    } KIdxStartEnd;

    vector<KIdxStartEnd> vec_k_idx_start_end;
    const U64 k_idx_max = pow(m, n*2);
    const U64 incrementer = [](const U64 incr) -> U64 {return (incr > 1000 ? 1000 : incr);}((k_idx_max / (U64)cpu_amount) + 1);
    // const U64 incrementer = 1000ull;
    for (U64 k_idx_start = 0ull; k_idx_start < k_idx_max; k_idx_start += incrementer) {
      U64 k_idx_end = k_idx_start + incrementer;
      if (k_idx_end > k_idx_max) {
        k_idx_end = k_idx_max;
      }
      vec_k_idx_start_end.emplace_back(k_idx_start, k_idx_end);
    }

    std::queue<U32> queue_number;

    const U64 size_vec = vec_k_idx_start_end.size();
    const U64 min_length = (cpu_amount > size_vec ? size_vec : cpu_amount);
    for (U32 i = 0; i < min_length; ++i) {
      ThreadData& thread_data = vec_thread_data[i];
      
      const KIdxStartEnd& k_idx_star_end = vec_k_idx_start_end[i];

      InputTypeOwn& inp_var = thread_data.var_input;
      inp_var.n = n;
      inp_var.m = m;
      inp_var.k_idx_start = k_idx_star_end.k_idx_start;
      inp_var.k_idx_end = k_idx_star_end.k_idx_end;

      ReturnTypeOwn& ret_var = thread_data.var_return;
      ret_var.map_len_cycle_to_count.clear();
      
      queue_number.push(i);
      // threads.emplace_back(start, &thread_data);
      // threads.emplace_back(thread_data.start, &thread_data);

      { // main to let start the thread to run
        std::lock_guard<std::mutex> lk(thread_data.mutex);
        thread_data.main_notify_thread = true;
        thread_data.thread_notify_main = false;
        thread_data.state = ThreadState::START;
      }
      thread_data.cond_var.notify_one();
    }

    U32 count_loops = 0;

    if (size_vec > cpu_amount) {
      // TODO: add the other stuff
      // assert(false && "Is not implemented yet!");
      for (U32 i_k_idx = min_length; i_k_idx < size_vec; ++i_k_idx) {
        U32 thread_nr = 0;
        while (true) {
          thread_nr = queue_number.front();
          queue_number.pop();
          ThreadData& thread_data = vec_thread_data[thread_nr];
          {
            std::unique_lock<std::mutex> lk(thread_data.mutex);
            if (thread_data.thread_notify_main) {
              count_loops = 0;

              const ReturnTypeOwn& ret_var = thread_data.var_return;
              const map<U32, U32>& map_len_cycle_to_count = ret_var.map_len_cycle_to_count;

              // TODO: need an extra fucntion for the accumulation!
              for (const auto& iter : map_len_cycle_to_count) {
                const U32 key = iter.first;
                const U32 value = iter.second;

                if (map_len_cycle_to_count_acc.find(key) == map_len_cycle_to_count_acc.end()) {
                  map_len_cycle_to_count_acc.emplace(key, 0u);
                }

                map_len_cycle_to_count_acc[key] += value;
              }

              break;
            } else {
              queue_number.push(thread_nr);
            }
          }

          count_loops += 1;
          if (count_loops >= queue_number.size()) {
            count_loops = 0;
            std::this_thread::yield();
          }
        }

        ThreadData& thread_data = vec_thread_data[thread_nr];
        
        const KIdxStartEnd& k_idx_star_end = vec_k_idx_start_end[i_k_idx];

        InputTypeOwn& inp_var = thread_data.var_input;
        inp_var.n = n;
        inp_var.m = m;
        inp_var.k_idx_start = k_idx_star_end.k_idx_start;
        inp_var.k_idx_end = k_idx_star_end.k_idx_end;

        ReturnTypeOwn& ret_var = thread_data.var_return;
        ret_var.map_len_cycle_to_count.clear();
        
        queue_number.push(thread_nr);

        { // main to let start the thread to run
          std::lock_guard<std::mutex> lk(thread_data.mutex);
          thread_data.main_notify_thread = true;
          thread_data.thread_notify_main = false;
          thread_data.state = ThreadState::START;
        }
        thread_data.cond_var.notify_one();
      }
    }

    while (queue_number.size() > 0) {
      const U32 thread_nr = queue_number.front();
      queue_number.pop();
      ThreadData& thread_data = vec_thread_data[thread_nr];
      {
        std::unique_lock<std::mutex> lk(thread_data.mutex);
        if (thread_data.thread_notify_main) {
          count_loops = 0;

          const ReturnTypeOwn& ret_var = thread_data.var_return;
          const map<U32, U32>& map_len_cycle_to_count = ret_var.map_len_cycle_to_count;

          // TODO: need an extra fucntion for the accumulation!
          for (const auto& iter : map_len_cycle_to_count) {
            const U32 key = iter.first;
            const U32 value = iter.second;

            if (map_len_cycle_to_count_acc.find(key) == map_len_cycle_to_count_acc.end()) {
              map_len_cycle_to_count_acc.emplace(key, 0u);
            }

            map_len_cycle_to_count_acc[key] += value;
          }

        } else {
          queue_number.push(thread_nr);
        }
      }

      count_loops += 1;
      if (count_loops >= queue_number.size()) {
        count_loops = 0;
        std::this_thread::yield();
      }
    }

    map_m_to_map_len_cycle_to_count.emplace(m, map_len_cycle_to_count_acc);
  }

  for (U32 i = 0; i < cpu_amount; ++i) {
    ThreadData& thread_data = vec_thread_data[i];
    { // stop all threads
      std::unique_lock<std::mutex> lk(thread_data.mutex);
      // main to let start the thread to run
      thread_data.main_notify_thread = true;
      thread_data.thread_notify_main = false;
      thread_data.state = ThreadState::END;
      
      thread_data.cond_var.notify_one();
    }
  }

  for (U32 i = 0; i < cpu_amount; ++i) {
    threads[i].join();
  }
}

int main(int argc, char* argv[]) {
  U32 cpu_amount = std::stoi(argv[4]);
  // U32 cpu_amount = std::thread::hardware_concurrency() - 1;

  const U32 n = std::stoi(argv[1]);
  // const U32 n = 2;
  map<U32, map<U32, U32>> map_m_to_map_len_cycle_to_count;

  const U32 m_min = std::stoi(argv[2]);
  const U32 m_max = std::stoi(argv[3]);
  // const U32 m_min = 1;
  // const U32 m_max = 20;
  calcCycleLengthAmountsMultiThreadingMainThread(n, m_min, m_max, cpu_amount, map_m_to_map_len_cycle_to_count);

  // cout << "map_m_to_map_len_cycle_to_count: " << map_m_to_map_len_cycle_to_count << endl;

  // TODO: make other functions for gethering interesting other sequences!
  vector<U32> sequence_m;
  vector<U32> sequence_a_m;

  for (const auto& map_len_cycle_to_count : map_m_to_map_len_cycle_to_count) {
    const auto& iter = map_len_cycle_to_count.second.rbegin();
    sequence_m.emplace_back(iter->first);
    sequence_a_m.emplace_back(iter->second);
  }

  cout << "sequence_m: " << sequence_m << endl;
  cout << "sequence_a_m: " << sequence_a_m << endl;

  return 0;
}
