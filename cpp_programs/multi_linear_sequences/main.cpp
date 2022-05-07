#include <iostream>
// import iostream;
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
#include "thread_data_generic.h"
#include "multi_linear_sequences.h"

// import thread_data_generic;

using namespace std;

using std::cout;
using std::endl;

// TODO: make a function for aggregating the return values from the threads!
void calcCycleLengthAmountsMultiThreadingMainThread(const U32 n, const U32 m_min, const U32 m_max, const U32 cpu_amount, std::map<U32, std::map<U32, U32>>& map_m_to_map_len_cycle_to_count) {
  std::vector<std::mutex> vec_mutex(cpu_amount);
  std::vector<std::condition_variable> vec_cond_var(cpu_amount);
  std::vector<ThreadData> vec_thread_data;
  for (U32 i = 0; i < cpu_amount; ++i) {
    vec_thread_data.emplace_back(vec_mutex[i], vec_cond_var[i], &calcCycleLengthAmounts);
    // vec_thread_data.emplace_back(vec_mutex[i], vec_cond_var[i], (void*)&calcCycleLengthAmounts);
  }

  std::vector<std::thread> threads;
  for (U32 i = 0; i < cpu_amount; ++i) {
    ThreadData& thread_data = vec_thread_data[i];
    std::thread th([&thread_data](){thread_data.start();});
    threads.push_back(std::move(th));
  }

  std::map<U32, U32> map_len_cycle_to_count_acc;
  for (U32 m = m_min; m <= m_max; ++m) {
    cout << "n: " << n << ", m: " << m << endl;

    map_len_cycle_to_count_acc.clear();

    typedef struct KIdxStartEnd_ {
      U64 k_idx_start;
      U64 k_idx_end;
    } KIdxStartEnd;

    std::vector<KIdxStartEnd> vec_k_idx_start_end;
    const U64 k_idx_max = pow(m, n*2);
    const U64 incrementer = [](const U64 incr) -> U64 {return (incr > 300 ? 300 : incr);}((k_idx_max / (U64)cpu_amount) + 1);
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
              const std::map<U32, U32>& map_len_cycle_to_count = ret_var.map_len_cycle_to_count;

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
          const std::map<U32, U32>& map_len_cycle_to_count = ret_var.map_len_cycle_to_count;

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
  // const U32 n = 2;
  const U32 n = std::stoi(argv[1]);

  const U32 m_min = std::stoi(argv[2]);
  const U32 m_max = std::stoi(argv[3]);
  
  U32 cpu_amount = std::stoi(argv[4]);
  
  std::map<U32, std::map<U32, U32>> map_m_to_map_len_cycle_to_count;
  // U32 cpu_amount = std::thread::hardware_concurrency() - 1;
  // const U32 m_min = 1;
  // const U32 m_max = 20;
  calcCycleLengthAmountsMultiThreadingMainThread(n, m_min, m_max, cpu_amount, map_m_to_map_len_cycle_to_count);

  // cout << "map_m_to_map_len_cycle_to_count: " << map_m_to_map_len_cycle_to_count << endl;

  // TODO: make other functions for gethering interesting other sequences!
  std::vector<U32> sequence_m;
  std::vector<U32> sequence_a_m;

  for (const auto& map_len_cycle_to_count : map_m_to_map_len_cycle_to_count) {
    const auto& iter = map_len_cycle_to_count.second.rbegin();
    sequence_m.emplace_back(iter->first);
    sequence_a_m.emplace_back(iter->second);
  }

  cout << "sequence_m: " << sequence_m << endl;
  cout << "sequence_a_m: " << sequence_a_m << endl;

  return 0;
}
