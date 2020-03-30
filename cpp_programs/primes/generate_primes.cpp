#include <assert.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>
#include <mutex>
#include <numeric>
#include <stdint.h>
#include <string>
#include <fstream>
#include <iterator>

#include "utils_primes.h"

using std::cout;
using std::endl;
using std::thread;
using std::ofstream;
using std::ifstream;
using std::ios;
using std::stoll;
using std::string;
using std::to_string;

void doSomething(int& a, std::mutex& mutex) {
    for (int i = 0; i < 10+a; ++i) {
        mutex.lock();
        cout << "a: " << a << endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        mutex.unlock();
        std::this_thread::yield();
    }
    a += 10 + a;
}

inline uint64_t multiply(const vector<uint64_t>& v) {
    auto binary_op_mult = [](int num1, int num2){ return num1 * num2; };
    return std::accumulate(v.begin(), v.end(), 1, binary_op_mult);
}

inline bool checkCoPrime(uint64_t n, const vector<uint64_t>& v) {
    size_t size = v.size();
    for (size_t i = 0; i < size; ++i) {
        const uint64_t k = v[i];
        if (n % k == 0 || k % n == 0) {
            return false;
        }
    }
    return true;
}

void createIncrementVector(const vector<uint64_t>& primes_part, vector<uint64_t>& increments) {
    uint64_t result = multiply(primes_part) + 1;
    // cout << "result: " << result << endl;

    vector<uint64_t> co_primes = {1};
    for (uint64_t i = 4; i <= result; ++i) {
        if (checkCoPrime(i, primes_part)) {
            co_primes.push_back(i);
        }
    }
    // cout << "co_primes: " << co_primes << endl;

    increments.resize(0);
    size_t size = co_primes.size();
    uint64_t co_prime_1 = co_primes[0];
    for (size_t i = 1; i < size; ++i) {
        uint64_t co_prime_2 = co_primes[i];
        increments.push_back(co_prime_2 - co_prime_1);
        co_prime_1 = co_prime_2;
    }
    // cout << "increments: " << increments << endl;
}

inline uint64_t intSqrt(const uint64_t n) {
  uint64_t n_1 = (n + 1) / 2;
  uint64_t n_2 = (n_1 + n / n_1) / 2;

  while (true) {
    uint64_t n_3 = (n_2 + n / n_2) / 2;
    if (n_3 == n_1) {
      return n_1;
    }
    n_1 = n_2;
    n_2 = n_3;
  }
}

void generatePrimes(vector<uint64_t>& primes, const uint64_t max_n) {
    const vector<uint64_t> start_primes = {2, 3, 5};
    primes.resize(0);
    primes.insert(primes.end(), start_primes.begin(), start_primes.end());

    const uint64_t increments[] = {4, 2};
    uint64_t starting_n = 7;

    uint64_t i_sqrt = 1;
    uint64_t i_sqrt_pow2 = i_sqrt * i_sqrt;
    uint64_t increment = 3;

    int increment_index = 0;
    for (uint64_t i = starting_n; i < max_n;) {
        if (i_sqrt_pow2 <= i) {
            i_sqrt += 1;
            i_sqrt_pow2 += increment;
            increment += 2;
        }

        bool is_prime = true;
        uint64_t p;
        for (uint64_t j = 0; (p = primes[j]) < i_sqrt; ++j) {
            if (i % p == 0) {
                is_prime = false;
                break;
            }
        }

        if (is_prime) {
            primes.push_back(i);
        }

        i += increments[increment_index];
        increment_index = (increment_index + 1) % 2;
    }
}

void generateNextPrimes(const vector<uint64_t>& primes, vector<uint64_t>& next_primes, const uint64_t start_n, const uint64_t end_n) {
    next_primes.resize(0);
    
    int increment_index = 1;
    uint64_t true_start_n = start_n;
    if (true_start_n % 2 == 0 && true_start_n % 3 == 0) {
        increment_index = 0;
        true_start_n += 1;
    } else if (true_start_n % 2 == 1 && true_start_n % 3 == 1) {
        increment_index = 0;
    } else if (true_start_n % 2 == 0 && true_start_n % 3 == 2) {
        true_start_n += 3;
    } else if (true_start_n % 2 == 1 && true_start_n % 3 == 0) {
        true_start_n += 2;
    } else if (true_start_n % 2 == 0 && true_start_n % 3 == 1) {
        true_start_n += 1;
    }

    uint64_t i_sqrt = intSqrt(true_start_n);
    uint64_t i_sqrt_pow2 = i_sqrt * i_sqrt;
    uint64_t increment = i_sqrt * 2 + 1;

    uint64_t increments[] = {4, 2};
    for (uint64_t i = true_start_n; i < end_n;) {
        if (i_sqrt_pow2 <= i) {
            i_sqrt += 1;
            i_sqrt_pow2 += increment;
            increment += 2;
            cout << "i_sqrt: " << i_sqrt << ", i_sqrt_pow2: " << i_sqrt_pow2 << endl;
        }
    
        uint64_t p;
        bool is_prime = true;
        for (uint64_t j = 0; (p = primes[j]) < i_sqrt; ++j) {
            if (i % p == 0) {
                is_prime = false;
                break;
            }
        }

        if (is_prime) {
            next_primes.push_back(i);
        }

        i += increments[increment_index];
        increment_index = (increment_index + 1) % 2;
    }
}

void generateNextPrimesMultithreaded(
        const vector<uint64_t>& primes,
        const unsigned int amount_rounds,
        const uint64_t numbers_range,
        vector<uint64_t>& primes_new) {
    unsigned concurrentThreadsSupported = std::thread::hardware_concurrency();
    
    primes_new.resize(0);
    vector<vector<uint64_t>> vector_primes(concurrentThreadsSupported);
    
    const uint64_t last_prime = primes[primes.size() - 1];
    uint64_t new_start_n = last_prime + 1;
    vector<thread> threads;
    for (unsigned int i = 0; i < concurrentThreadsSupported; ++i) {
        thread t(generateNextPrimes, std::ref(primes), std::ref(vector_primes[i]), new_start_n, new_start_n + numbers_range);
        new_start_n += numbers_range;
        threads.push_back(std::move(t));
    }

    for (unsigned int round = 0; round < amount_rounds; ++round) {
        for (unsigned int i = 0; i < concurrentThreadsSupported; ++i) {
            thread& t = threads[i];
            t.join();
            vector<uint64_t>& v = vector_primes[i];
            primes_new.insert(primes_new.end(), v.begin(), v.end());
            thread t2(generateNextPrimes, std::ref(primes), std::ref(v), new_start_n, new_start_n + numbers_range);
            new_start_n += numbers_range;
            threads[i] = std::move(t2);
        }
    }

    for (unsigned int i = 0; i < concurrentThreadsSupported; ++i) {
        thread& t = threads[i];
        t.join();
        vector<uint64_t>& v = vector_primes[i];
        primes_new.insert(primes_new.end(), v.begin(), v.end());
    }
}

int main(int argc, char* argv[]) {
    vector<uint64_t> primes;

    uint64_t n_max = 1000000ULL;
    if (argc >= 2) {
        n_max = stoll(argv[1]);
    }

    /*
    // only for single core use!
    cout << "Generating primes until n_max: " << n_max << endl;
    generatePrimes(primes, n_max);
    cout << "Finished." << endl;

    string file_path = "/tmp/primes_n_max_" + to_string(n_max) + ".dat";
    ofstream fout_old(file_path, ios::out | ios::binary);
    uint64_t size_primes = primes.size();
    fout_old.write((char*)&size_primes, sizeof(uint64_t));
    fout_old.write((char*)&primes[0], primes.size() * sizeof(uint64_t));
    fout_old.close();
    */

    generatePrimes(primes, 10000);
    vector<uint64_t> primes_new;
    uint64_t increment = 10000;
    for (int i = 0; i < 50; ++i) {
        cout << "i: " << i << endl;
        generateNextPrimesMultithreaded(primes, 5, increment, primes_new);

        primes.insert(primes.end(), primes_new.begin(), primes_new.end());
        const uint64_t last_num = primes[primes.size()-1];

        increment += 10000;

        if (last_num >= n_max) {
            break;
        }
    }

    for (uint64_t i = primes.size() - 1;; --i) {
        if (primes[i] <= n_max) {
            primes.resize(i+1);
            break;
        }
    }

    ofstream fout("/tmp/primes_multithreaded_n_max_" + to_string(n_max) + ".dat", ios::out | ios::binary);
    uint64_t size = primes.size();
    cout << "size: " << size << endl;
    fout.write((char*)&size, sizeof(uint64_t));
    fout.write((char*)&primes[0], primes.size() * sizeof(uint64_t));
    fout.close();

    return 0;
}
