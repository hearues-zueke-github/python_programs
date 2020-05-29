//
// Created by doublepmcl on 06.06.19.
//

#ifndef SIMPLEVECTORPRINTS_POLYNOME_H
#define SIMPLEVECTORPRINTS_POLYNOME_H

#include <vector>

#include "Factors.h"

using namespace std;

class Polynome {
private:
    friend class Polynomial;
    vector<int32_t> _factors;
    void shortenPolynome(const vector<int32_t>& src, vector<int32_t>& dst);
    void shiftFactors(int32_t factor);
    void checkZerosFactors();
    void multiplyWithInt(int32_t factor);
public:
    Polynome(const vector<int32_t>& factors);
    Polynome(const Polynome& obj);
    virtual ~Polynome();

    const vector<int32_t>& getFactors() const;

    friend ostream& operator<<(ostream& os, const Polynome& obj);
    Polynome& operator=(const Polynome& rhs);
    Polynome operator+(const Polynome& rhs);
    Polynome operator-(const Polynome& rhs);
    friend Polynome operator*(const Polynome& lhs, const int32_t rhs);
    friend Polynome operator*(const int32_t lhs, const Polynome& rhs);
    friend Polynome operator*(const Polynome& lhs, const Polynome& rhs);
    friend bool operator==(const Polynome& lhs, const Polynome& rhs);
//    Polynome operator*(const int32_t rhs);
//    Polynome operator()(const int32_t rhs);
};


#endif //SIMPLEVECTORPRINTS_POLYNOME_H
