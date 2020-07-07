#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdint>
#include <iomanip>

using std::cout;
using std::endl;
using std::ostream;
using std::ofstream;
using std::string;
using std::stringstream;

string intToHex(uint64_t n) {
  if (n == 0ull) {
    return "0x0000000000000000";
  }

  stringstream s;
  s << std::showbase << std::hex << std::uppercase << n;
  string st = s.str();
  st = st.substr(2, st.length() - 2);
  s.str(string());
  s << std::setfill('0') << std::setw(16) << st;
  return "0x"+s.str();
}

struct Bits64 {
  uint8_t bits[64];

  Bits64(uint64_t num) {
    for (int i = 0; i < 64; ++i) {
      bits[i] = num % 2;
      num /= 2;
    }
  }

  void writeBitsStringU64(ofstream& f) {
    for (int i = 63; i >= 0; --i) {
      f << (int)bits[i];
    }
    f << "\n";
  }
};

struct HashIterator {
  uint64_t I = 0;
  uint64_t x = 0;
  uint64_t y = 0;
  uint64_t z = 0;
  uint64_t w = 0;

  void doNextCalcStep_old() {
    ++I;
    x += I + (x >> 1);
    y = (x ^ y) + (x >> 1);
    z = x * I + (z << 1) + (I ^ z);
    w = (z << 1) ^ (y >> 1);
  }

  void doNextCalcStep() {
    ++I;
    x = ((x << 1) ^ I) + (x >> 1);
    y = ((y << 1) ^ x) + I;
    z = (I ^ z) + (z << 1) + (x < y ? x : -y);
    w = ((z >> 1) ^ (y << (z > y))) + (w & I);
  }

  void writeToFiles(ofstream& fI, ofstream& fX, ofstream& fY, ofstream& fZ, ofstream& fW) {
    Bits64(I).writeBitsStringU64(fI);
    Bits64(x).writeBitsStringU64(fX);
    Bits64(y).writeBitsStringU64(fY);
    Bits64(z).writeBitsStringU64(fZ);
    Bits64(w).writeBitsStringU64(fW);
  }
};

ostream& operator<<(ostream& os, const HashIterator& obj) {
  os << "obj.I: " << intToHex(obj.I);
  os << ", obj.x: " << intToHex(obj.x);
  os << ", obj.y: " << intToHex(obj.y);
  os << ", obj.z: " << intToHex(obj.z);
  os << ", obj.w: " << intToHex(obj.w);
  return os;
}

ostream& operator<<(ostream& os, const Bits64& obj) {
  auto& bits = obj.bits;
  os << "Bits64(";
  for (int i = 63; i >= 0; --i) {
    os << (int)bits[i] << ",";
  }
  os << ")";

  return os;
}

int main(int argc, char* argv[]) {
  ofstream ofVarI;
  ofstream ofVarX;
  ofstream ofVarY;
  ofstream ofVarZ;
  ofstream ofVarW;

  ofVarI.open("values_I.txt");
  ofVarX.open("values_X.txt");
  ofVarY.open("values_Y.txt");
  ofVarZ.open("values_Z.txt");
  ofVarW.open("values_W.txt");

  // Bits64 b64 = Bits64(0x0123456789ABCDEFULL);
  // cout << "b64: " << b64 << endl;

  int32_t maxLoops = 1000;

  HashIterator hi{
    0x0000000001000000ULL,
    0x0000000000000000ULL,
    0x0000000000000000ULL,
    0x0000000000000000ULL,
    0x0000000000000000ULL
  };
  hi.writeToFiles(ofVarI, ofVarX, ofVarY, ofVarZ, ofVarW);

  for (int i = 0; i < maxLoops; ++i) {
    hi.doNextCalcStep();
    cout << "hi: " << hi << endl;
    hi.writeToFiles(ofVarI, ofVarX, ofVarY, ofVarZ, ofVarW);
  }

  ofVarI.close();
  ofVarX.close();
  ofVarY.close();
  ofVarZ.close();
  ofVarW.close();
}
