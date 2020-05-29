//
// Created by doublepmcl on 14.06.19.
//

#ifndef SEQUENCESGENERATOR_COMBINATIONS_H
#define SEQUENCESGENERATOR_COMBINATIONS_H

#include <cstdint>
#include <vector>
#include <ostream>

using namespace std;

class Combinations {
private:
  size_t _m;
  size_t _n;
  vector<uint32_t> _vals;
  size_t _iter;
//  size_t _max_iters;
  size_t ownPow(size_t b, size_t e);
public:
  Combinations(int m, int n);
  Combinations(const Combinations& obj);
  virtual ~Combinations();

  friend ostream& operator<<(ostream& os, const Combinations& obj);

  void clearVector();
  bool nextCombo();
};


#endif //SEQUENCESGENERATOR_COMBINATIONS_H
