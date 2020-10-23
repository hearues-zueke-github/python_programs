//
// Created by doublepmcl on 06.06.19.
//

#ifndef SIMPLEVECTORPRINTS_POLYNOMIAL_H
#define SIMPLEVECTORPRINTS_POLYNOMIAL_H

using namespace std;

#include <iostream>
#include <vector>

#include "Polynome.h"

using namespace std;

class Polynomial {
private:
    vector<Polynome> _polynomes;
public:
    Polynomial(const vector<Polynome>& polynomes);
    Polynomial(const Polynomial& obj);
    virtual ~Polynomial();

    void multiplyPolynomialItself();

    friend ostream& operator<<(ostream& os, const Polynomial& obj);
};


#endif //SIMPLEVECTORPRINTS_POLYNOMIAL_H
