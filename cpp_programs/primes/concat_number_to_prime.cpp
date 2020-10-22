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
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#include <string>

#include "utils_primes.h"

using std::string;
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

// fast and dirty prime number check
bool isPrime(const uint64_t n, const vector<uint64_t>& primes) {
    const size_t size = primes.size();

    // calc the max p to which it should be checked!
    const uint64_t max_p = intSqrt(n);

    for (size_t i = 0; i < size; ++i) {
        const uint64_t p = primes[i];
        if (p >= max_p) {
            break;
        }

        if (n % p == 0) {
            return false;
        }
    }

    return true;
}

int main(int argc, char* argv[]) {
    unsigned concurrentThreadsSupported = std::thread::hardware_concurrency();
    cout << "concurrentThreadsSupported: " << concurrentThreadsSupported << endl;

    struct passwd *pw = getpwuid(getuid());
    const string homedir = string(pw->pw_dir);
    std::ifstream fin(homedir+string("/Downloads/prime_numbers_data.dat"), std::ios_base::binary);

    uint64_t size_read;
    fin.read(reinterpret_cast<char*>(&size_read), sizeof(uint64_t));
    cout << "size_read: " << size_read << endl;

    vector<uint64_t> primes;
    primes.resize(size_read);
    fin.read(reinterpret_cast<char*>(&primes[0]), size_read*sizeof(uint64_t));
    cout << "primes.size(): " << primes.size() << endl;
    cout << "primes[primes.size()-1]:" << primes[primes.size()-1] << endl;

    uint64_t max_k = 0;
    uint64_t max_j = 0;
    uint64_t max_n = 0;
    uint64_t max_n_concat = 0;

    const uint64_t b = 5;
    vector<vector<uint64_t>> found_values;

    for (uint64_t n = 1; n < 500000ULL; ++n) {
        if (n % 100000 == 0) {
            cout << "n: " << n << endl;
        }
        uint64_t found_k = 0;
        uint64_t found_j = 0;
        uint64_t found_n = 0;
        uint64_t found_n_concat = 0;

        uint64_t b_pow = 1;
        for (uint64_t k = 1; k < 9; ++k) {
            b_pow *= b;
            const uint64_t n2 = n * b_pow;
            for (uint64_t j = 1; j < b_pow; ++j) {
                const uint64_t n_concat = n2 + j;
                if (isPrime(n_concat, primes)) {
                    found_k = k;
                    found_j = j;
                    found_n = n;
                    found_n_concat = n_concat;
                    goto Break_Loops;
                }
            }
        }

        continue;
        Break_Loops:
        if (max_k < found_k) {
            max_k = found_k;
            max_j = found_j;
            max_n = found_n;
            max_n_concat = found_n_concat;
            vector<uint64_t> values{found_k, found_j, found_n, found_n_concat};
            cout << "values found: " << values << endl;
            found_values.push_back(values);
        }
    }

    cout << "found_values: " << found_values << endl;

    return 0;
}
