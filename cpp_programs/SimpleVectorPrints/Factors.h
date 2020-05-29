//
// Created by doublepmcl on 06.06.19.
//

#ifndef SIMPLEVECTORPRINTS_FACTORS_H
#define SIMPLEVECTORPRINTS_FACTORS_H


#include <cstdint>
#include <ostream>

using namespace std;

class Factors {
private:
    friend class Polynome;
    int32_t _a;
    uint32_t _b;
public:
    Factors(const int32_t a, const uint32_t b);
    Factors(const Factors& obj);
    virtual ~Factors();

    int32_t getA();
    uint32_t getB();

    friend ostream& operator<<(ostream& os, const Factors& obj);

    Factors& operator= (Factors const& rhs);
    Factors operator+=(const Factors& rhs);
};


#endif //SIMPLEVECTORPRINTS_FACTORS_H
