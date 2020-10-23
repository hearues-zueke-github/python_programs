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
#include <math.h>

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

// primes is changed in the function!!!
void generateNextPrimesMultithreadedBetter(
        vector<uint64_t>& primes,
        const uint64_t n_max) {
    uint64_t starting_n = 100000;
    generatePrimes(primes, starting_n);

    vector<uint64_t> primes_copy(primes);

    const unsigned concurrentThreadsSupported = std::thread::hardware_concurrency();
    vector<vector<uint64_t>> vector_primes(concurrentThreadsSupported);

    vector<uint64_t> n_values;
    n_values.push_back(starting_n);

    double n_splits = (double)concurrentThreadsSupported;
    for (uint64_t i = 1; i < concurrentThreadsSupported; ++i) {
        const uint64_t next_n = (uint64_t)pow((double)i / n_splits * pow(n_max, 3. / 2) + pow(starting_n, 3. / 2) * (1. - (double)i / n_splits), 2. / 3);
        n_values.push_back(next_n);
    }
    n_values.push_back(n_max);

    cout << "n_values: " << n_values << endl;

    vector<thread> threads;
    for (unsigned int i = 0; i < concurrentThreadsSupported; ++i) {
        thread t(generateNextPrimes, std::ref(primes_copy), std::ref(vector_primes[i]), n_values[i], n_values[i + 1]);
        threads.push_back(std::move(t));

    }

    for (unsigned int i = 0; i < concurrentThreadsSupported; ++i) {
        thread& t = threads[i];
        t.join();
        vector<uint64_t>& v = vector_primes[i];
        primes.insert(primes.end(), v.begin(), v.end());
    }

    for (uint64_t i = primes.size() - 1;; --i) {
        if (primes[i] <= n_max) {
            primes.resize(i+1);
            break;
        }
    }
}

int main(int argc, char* argv[]) {
    vector<uint64_t> primes;

    uint64_t n_max = 1000000ULL;
    if (argc >= 2) {
        n_max = stoll(argv[1]);
    }

    // generateNextPrimesMultithreadedBetter(primes, n_max);

    generatePrimes(primes, 10000);
    vector<uint64_t> primes_new;
    uint64_t increment = 100000;
    for (int i = 0; i < 50; ++i) {
        cout << "i: " << i << endl;
        generateNextPrimesMultithreaded(primes, 5, increment, primes_new);

        primes.insert(primes.end(), primes_new.begin(), primes_new.end());
        const uint64_t last_num = primes[primes.size()-1];

        increment += 100000;

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
    // ofstream fout("/tmp/primes_multithreaded_better_n_max_" + to_string(n_max) + ".dat", ios::out | ios::binary);
    uint64_t size = primes.size();
    cout << "size: " << size << endl;
    fout.write((char*)&size, sizeof(uint64_t));
    fout.write((char*)&primes[0], primes.size() * sizeof(uint64_t));
    fout.close();

    return 0;
}
