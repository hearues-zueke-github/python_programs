#include <iostream>
#include <sstream>
#include <string>
#include <cstdint>
#include <iomanip>

std::string intToHex(uint64_t n) {
    if (n == 0ull) {
        return "0x0000000000000000";
    }
    std::stringstream s;
    s << std::showbase << std::hex << std::uppercase << n;
    std::string st = s.str();
    // std::string::replace(st.begin(), st.end(), "0X", "0x");
    // std::cout << "st: " << st << std::endl;
    st = st.substr(2, st.length() - 2);
    s.str(std::string());
    s << std::setfill('0') << std::setw(16) << st;
    return "0x"+s.str();
    // st.replace(st.find("0X"), 0, "");
    // return st;
}

int main(int argc, char* argv[]) {
    // std::ios_base::fmtflags f(std::cout.flags());

    uint64_t maxLoops = 100000000ull;

    uint64_t x = 0;
    uint64_t y = 0;
    uint64_t z = 0;
    uint64_t w = 0;
    uint64_t I = 0;
    for (; I < maxLoops; I++) {
    // for (uint64_t I = 0; I < 100000001ULL; I++) {
        x += I + (x >> 1);
        y = (x ^ y) + (x >> 1);
        z = x * I + (z << 1) + (I ^ z);
        w = (z << 1) ^ (y >> 1);
        if (I % 1000 == 0) {
            std::cout << "I: " << intToHex(I) << ", x: " << intToHex(x) << ", y: " << intToHex(y) << ", z: " << intToHex(z) << ", w: " << intToHex(w) << std::endl;
        }
    }
    
    std::cout << "I: " << intToHex(I) << ", x: " << intToHex(x) << ", y: " << intToHex(y) << ", z: " << intToHex(z) << ", w: " << intToHex(w) << std::endl;
    
    // std::cout.flags(f);
}
