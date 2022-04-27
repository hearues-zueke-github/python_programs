#pragma once

#include "utils.h"

namespace RandomGen {
  template<typename T>
  class RandomGenInt {
  private:
    std::seed_seq seeder_;
    std::mt19937 rng_;
    std::uniform_int_distribution<T> dist_;

  public:
    RandomGenInt(const std::vector<uint32_t>& seed, const T minVal, const T maxVal) :
        seeder_(seed.begin(), seed.end()), rng_(seeder_), dist_(minVal, maxVal)
    {
    }

    RandomGenInt(const std::vector<uint32_t>& seed) :
        seeder_(seed.begin(), seed.end()), rng_(seeder_), dist_(std::numeric_limits<T>::min(), std::numeric_limits<T>::max())
    {
    }

    inline T nextVal() {
      return dist_(rng_);
    }

    void resetSeed() {
      rng_.seed(seeder_);
    }

    inline void generateNextVals(const size_t amount, vector<T>& vec) {
      vec.resize(amount);
      for (size_t i = 0; i < amount; ++i) {
        vec[i] = nextVal();
      }
    }
  };

  template<typename T>
  class RandomGenReal {
  private:
    std::seed_seq seeder_;
    std::mt19937 rng_;
    std::uniform_real_distribution<T> dist_;
  public:
    RandomGenReal(const std::vector<uint32_t>& seed, const T minVal, const T maxVal) :
        seeder_(seed.begin(), seed.end()), rng_(seeder_), dist_(minVal, maxVal)
    {
    }

    RandomGenReal(const std::vector<uint32_t>& seed) :
        seeder_(seed.begin(), seed.end()), rng_(seeder_), dist_(std::numeric_limits<T>::min(), std::numeric_limits<T>::max())
    {
    }

    inline T nextVal() {
      return dist_(rng_);
    }

    void resetSeed() {
      rng_.seed(seeder_);
    }

    inline void generateNextVals(const size_t amount, vector<T>& vec) {
      vec.resize(amount);
      for (size_t i = 0; i < amount; ++i) {
        vec[i] = nextVal();
      }
    }
  };
}
