#include <iostream>
#include <fstream>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>

// #include <vector>
#include <iomanip>
#include <ostream>
#include <sstream>

#include "utils.h"
#include "COLORS.h"

using namespace std;

using i8 = int8_t;
using ui8 = uint8_t;

ui8 get4BitLog(ssize_t a) {
  ui8 log_num = 0;
  while (a > 0) {
    a >>= 4;
    ++log_num;
  }
  return log_num;
}

// template<typename T>
// void outputHex(const vector<T>& buffer, const ssize_t size) {
void outputHex(const vector<i8>& buffer, const ssize_t size) {
  const i8* data = buffer.data();
  const ssize_t per_row = 0x10;
  const ssize_t rows = size / per_row;
  const ssize_t last_row_amount = (size % per_row > 0) ? (size % per_row) : 0;

  // stringstream ss_first;
  auto setStringStream = [](stringstream& ss, int width) {
    ss << hex << uppercase << setw(width) << setfill('0');
  };

  ui8 hex_length = get4BitLog(size-1);
  uint64_t mask = (1<<(4*hex_length))-1;
  printf("mask: 0x%016lX\n", mask);

  for (ssize_t j = 0; j < rows; ++j) {
    const i8* row = data + j * per_row;
    // cout << "row:"; // << j << ": ";
    stringstream ss_2;
    setStringStream(ss_2, hex_length);
    ss_2 << ((j*per_row) & mask);
    cout << "" << ss_2.str() << ":";
    for (ssize_t i = 0; i < per_row; ++i) {
      stringstream ss;
      setStringStream(ss, 2);
      ss << (row[i] & 0xFF);
      cout << " " << ss.str();
    }
    cout << endl;
  }

  if (last_row_amount > 0) {
    const i8* row = data + per_row * rows;
    // cout << "row:"; // << rows << ": ";
    stringstream ss_2;
    setStringStream(ss_2, hex_length);
    ss_2 << ((rows*per_row) & mask);
    cout << "" << ss_2.str() << ":";

    for (ssize_t i = 0; i < last_row_amount; ++i) {
      stringstream ss;
      setStringStream(ss, 2);
      ss << (row[i] & 0xFF);
      cout << " " << ss.str();
    }
    cout << endl;
  }
}

int main(int argc, char* argv[]) {

  if (argc < 2) {
    cout << "Not enough arguments!" << endl;
    cout << "usage: ./<program> <filepath>" << endl;
    return -1;
  }

  char* filepath = argv[1];
  cout << "filepath: " << filepath << endl;
  struct stat s;
  if ( stat(filepath, &s) == 0 ) {
    if ( s.st_mode & S_IFDIR ) {
      cout << "Found a directory! Needed a file! exit..." << endl;
      return -3;
    } else if ( s.st_mode & S_IFREG ) {
      cout << "filepath: " << filepath << " is a file!" << endl;
    } else {
      cout << "Something else found! exit..." << endl;
      return -1;
    }
  } else {
    cout << "Something else went wrong! exit..." << endl;
    return -2;
  }

  ifstream file(filepath, ios::binary | ios::ate);
  streamsize size = file.tellg();
  file.seekg(0, ios::beg);

  vector<char> buffer(size);
  size_t i = 0;
  if (file.read(buffer.data(), size))
  {
      streamsize s = file.gcount();
      cout << "i:" << i << ", s: " << s << ", buffer:" << endl;
      // vector<int8_t> buffer2;
      vector<int8_t> buffer2;
      copyValues(buffer, buffer2);
      outputHex(buffer2, size);
      cout << endl;
      ++i;
  }

  return 0;
}
