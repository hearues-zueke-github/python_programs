#ifndef UTILS_TETRIS_H
#define UTILS_TETRIS_H

#include <random>
#include <set>
#include <string>

template<typename Iter, typename RandomGenerator>
Iter select_randomly(Iter start, Iter end, RandomGenerator& g);

template<typename Iter>
Iter select_randomly(Iter start, Iter end);

#endif // UTILS_TETRIS_H
