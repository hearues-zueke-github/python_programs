//
// Created by doublepmcl on 06.06.19.
//

#ifndef SIMPLEVECTORPRINTS_NUMBERS_H
#define SIMPLEVECTORPRINTS_NUMBERS_H

#include <vector>
#include <cstdint>
#include <ostream>

using namespace std;

class Numbers {
private:
    vector<uint64_t> _nums;
public:
    Numbers();
    Numbers(const vector<uint64_t>& nums);
    Numbers(const Numbers& obj);
    virtual ~Numbers();

    vector<uint64_t>& getNums();

    friend ostream& operator<<(ostream& os, const Numbers& obj);
};


#endif //SIMPLEVECTORPRINTS_NUMBERS_H
