#include <assert.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>
#include <mutex>
#include <numeric>
#include <stdint.h>
// #include <stdio.h>
#include <fstream>
#include <iterator>

#include "utils_primes.h"

using std::cout;
using std::endl;
using std::thread;
using std::ofstream;
using std::ifstream;
using std::ios;

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

void generatePrimes(vector<uint64_t>& primes, const uint64_t max_n) {
    const vector<uint64_t> start_primes = {2, 3, 5};
    primes.resize(0);
    primes.insert(primes.end(), start_primes.begin(), start_primes.end());

    uint64_t primes_size = primes.size();
    uint64_t increments[] = {4, 2};
    int increment_index = 0;
    for (uint64_t i = 7; i < max_n;) {
        bool is_prime = true;
        for (uint64_t j = 0; j < primes_size; ++j) {
            uint64_t p = primes[j];
            uint64_t div = i / p;

            if (div < p) {
                break;
            }

            if (i % p == 0) {
                is_prime = false;
                break;
            }
        }

        if (is_prime) {
            primes.push_back(i);
            ++primes_size;
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

    // find the prime size, which is needed for calculating the next prime starting from
    // start_n number!
    const uint64_t smallest_prime_test = sqrt(start_n) + 1ULL;
    const size_t primes_size = primes.size();
    uint64_t primes_size_check = 0;
    for (; primes_size_check < primes_size; ++primes_size_check) {
        if (primes[primes_size_check] >= smallest_prime_test) {
            break;
        }
    }
    
    if (primes[primes_size_check] < smallest_prime_test) {
        assert(0);
    }
    uint64_t biggest_prime_for_check = primes[primes_size_check];
    ++primes_size_check;

    uint64_t increments[] = {4, 2};
    for (uint64_t i = true_start_n; i < end_n;) {
        if (i / biggest_prime_for_check > biggest_prime_for_check) {
            biggest_prime_for_check = primes[primes_size_check];
            ++primes_size_check;
        }

        // bool is_prime = true;
        for (uint64_t j = 0; j < primes_size_check; ++j) {
            uint64_t p = primes[j];
            if (i % p == 0) {
                goto BREAK_FOR_LOOP;
                // is_prime = false;
                // break;
            }
        }

        // if (is_prime) {
        next_primes.push_back(i);
        // }
        BREAK_FOR_LOOP:
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
    cout << "concurrentThreadsSupported: " << concurrentThreadsSupported << endl;
    
    primes_new.resize(0);
    vector<vector<uint64_t>> vector_primes(concurrentThreadsSupported);
    
    const uint64_t last_prime = primes[primes.size() - 1];
    uint64_t new_start_n = last_prime + 1;
    vector<thread> threads;
    for (unsigned int i = 0; i < concurrentThreadsSupported; ++i) {
        thread t(generateNextPrimes, std::ref(primes), std::ref(vector_primes[i]), new_start_n, new_start_n + numbers_range);
        new_start_n += numbers_range;
        cout << "i: " << i << ", new_start_n: " << new_start_n << endl;
        threads.push_back(std::move(t));
    }

    for (unsigned int round = 0; round < amount_rounds; ++round) {
        cout << "round: " << round << endl;

        for (unsigned int i = 0; i < concurrentThreadsSupported; ++i) {
            thread& t = threads[i];
            t.join();
            vector<uint64_t>& v = vector_primes[i];
            // cout << "i: " << i << ", v: " << v << endl;
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

void generatePrimesIncrements(const size_t start_primes_amount, vector<uint64_t>& primes, const uint64_t max_n) {
    generatePrimes(primes, 100);
    primes.resize(start_primes_amount);

    vector<uint64_t> increments;
    createIncrementVector(primes, increments);

    uint64_t primes_size = primes.size();
    size_t max_prime_index = start_primes_amount - 1;
    size_t increment_index = 1;
    const size_t size_increments = increments.size();
    for (uint64_t i = increments[0] + 1; i < max_n;) {
        const uint64_t p = primes[max_prime_index - 1];
        uint64_t div = i / p;
        if (div > p) {
            ++max_prime_index;
        }

        for (uint64_t j = start_primes_amount; j < max_prime_index; ++j) {
            if (i % primes[j] == 0) {
                goto LABEL_NOT_PRIME;
            }
        }

        primes.push_back(i);
        ++primes_size;

        LABEL_NOT_PRIME:
        i += increments[increment_index];
        increment_index = (increment_index + 1) % size_increments;
    }
}

int main(int argc, char* argv[]) {
    vector<uint64_t> primes;

    // // useful for reading the file with pre-calculated prime numbers!
    // std::ifstream fin("/tmp/primes_data_test_2.dat", std::ios_base::binary);
    // uint64_t size_read;
    // fin.read(reinterpret_cast<char*>(&size_read), sizeof(uint64_t));
    // primes.resize(size_read);
    // fin.read(reinterpret_cast<char*>(&primes[0]), size_read*sizeof(uint64_t));

    generatePrimes(primes, 100000);

    // // useful for writting the prime numbers to the file!
    // ofstream fout("/tmp/primes_data.dat", ios::out | ios::binary);
    // uint64_t size = primes.size();
    // fout.write((char*)&size, sizeof(uint64_t));
    // fout.write((char*)&primes[0], primes.size() * sizeof(uint64_t));
    // fout.close();
    
    vector<uint64_t> primes_new;
    for (int i = 0; i < 4; ++i) {
        generateNextPrimesMultithreaded(primes, 10, 100000, primes_new);

        cout << "primes_new.size(): " << primes_new.size() << endl;
        primes.insert(primes.end(), primes_new.begin(), primes_new.end());
        cout << "new primes:" << endl;
        cout << "primes.size(): " << primes.size() << endl;
        cout << "primes[primes.size()-1]: " << primes[primes.size()-1] << endl;
    }

    ofstream fout("/tmp/primes_data_test_3.dat", ios::out | ios::binary);
    uint64_t size = primes.size();
    fout.write((char*)&size, sizeof(uint64_t));
    fout.write((char*)&primes[0], primes.size() * sizeof(uint64_t));
    fout.close();

    return 0;
}
