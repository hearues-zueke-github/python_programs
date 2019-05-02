#include <assert.h>
#include <stdio.h>
#include <sys/stat.h>

#include <cstdint>
#include <cstdlib>

#include <iostream>
#include <fstream>
#include <sstream>

#include <memory>
#include <string>
#include <thread>
#include <typeinfo>
#include <type_traits>
#include <vector>

#include <boost/lexical_cast.hpp>
#include <boost/multiprecision/gmp.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_complex.hpp>

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

using std::cout;
using std::cin;
using std::endl;
using std::string;

using namespace boost::multiprecision;

template <class T>
string getTypName ()
{
    typedef typename std::remove_reference<T>::type TR;
    std::unique_ptr<char, void(*)(void*)> own
           (
                nullptr,
                std::free
           );
    std::string r = own != nullptr ? own.get() : typeid(TR).name();
    if (std::is_const<TR>::value)
        r += " const";
    if (std::is_volatile<TR>::value)
        r += " volatile";
    if (std::is_lvalue_reference<T>::value)
        r += "&";
    else if (std::is_rvalue_reference<T>::value)
        r += "&&";
    return r;
}

// typedef number<cpp_dec_float<20>> float_own;
typedef number<cpp_bin_float<50>> bin_float_own;
typedef cpp_complex<50> cplx_own;

template <typename T> std::string type_name();

void testingBoostFloatComplex () {
  cplx_own c1(2, 3);
  cplx_own c2(3, 7);

  cout << "c1: " << c1 << endl;
  cout << "c2: " << c2 << endl;
  
  cplx_own c_plus = c1+c2;
  cout << "c_plus: " << c_plus << endl;
  cplx_own c_minus = c1-c2;
  cout << "c_minus: " << c_minus << endl;
  cplx_own c_mul = c1*c2;
  cout << "c_mul: " << c_mul << endl;
  cplx_own c_div = c1/c2;
  cout << "c_div: " << c_div << endl;

  cout << "real_part of c1: " << real(c1)/7 << endl;
  cout << "imag_part of c1: " << imag(c1)/7 << endl;

  // for getting the type_name of a type as string
  cout << "typename of cplx_own: " << typeid(cplx_own::value_type).name() << endl;
}

inline bool fileExists (const string& name) {
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0); 
}

void readAllBytes(char const* filename, std::vector<char>& result) {
  std::ifstream ifs(filename, std::ios::binary|std::ios::ate);
  std::ifstream::pos_type pos = ifs.tellg();
  // std::vector<char> result(pos);
  result.reserve(pos);
  ifs.seekg(0, std::ios::beg);
  ifs.read(&result[0], pos);
  // return result;
}

int main (int argc, char* argv[]) {
  cout << std::setprecision(std::numeric_limits<typename cplx_own::value_type>::digits10);

  assert (argc >= 2);

  string filePath = argv[1];
  bool exists = fileExists(filePath);
  cout << "exists: " << exists << endl;
  assert(exists);

  // std::vector<char> vecParams;
  // readAllBytes(argv[1], vecParams);
  // cout << "vecParams.size(): " << vecParams.size() << endl;

  std::ifstream f(filePath, std::ios::binary);
  std::vector<char> vecParams(std::istreambuf_iterator<char>{f}, {});
  cout << "vecParams.size(): " << vecParams.size() << endl;

  string paramsFile = &vecParams[0];
  cout << "paramsFile: " << paramsFile << endl;

  // TODO: convert string paramsFile into h, w, x0, y0 and w1 values!

  string pw;
  string ph;
  string px0;
  string py0;
  string pw1;

  std::vector<string> lines;
  boost::split(lines, paramsFile, boost::is_any_of("\n"));
  
  std::vector<string> ps;
  boost::split(ps, lines[0], boost::is_any_of(","));
  for (unsigned i = 0; i < ps.size(); ++i) {
    cout << "i: " << i << ", ps["<< i << "]: " << ps[i] << endl;
  }

  assert (ps.size() == 6);

  // return 0;

  unsigned h = std::stoi(ps[0]); // 0x12;
  unsigned w = std::stoi(ps[1]); // 0x18;
  bin_float_own y0(ps[2]);
  bin_float_own x0(ps[3]);
  bin_float_own w1(ps[4]);
  string outputFilePath = ps[5];
  // bin_float_own h1("0.0125");

  bin_float_own h1 = bin_float_own(h) / bin_float_own(w) * w1;

  bin_float_own const_2("2");

  bin_float_own x1 = x0 - w1 / const_2;
  bin_float_own y1 = y0 - h1 / const_2;

  bin_float_own x2 = x0 + w1 / const_2;
  bin_float_own y2 = y0 + h1 / const_2;

  bin_float_own dx = (x2 - x1) / w;
  bin_float_own dy = (y2 - y1) / h;

  cout << "w: " << w << endl;
  cout << "h: " << h << endl;

  cout << "w1: " << w1 << endl;
  cout << "h1: " << h1 << endl;

  cout << "x0: " << x0 << endl;
  cout << "y0: " << y0 << endl;

  cout << "x1: " << x1 << endl;
  cout << "y1: " << y1 << endl;

  cout << "x2: " << x2 << endl;
  cout << "y2: " << y2 << endl;

  cout << "dx: " << dx << endl;
  cout << "dy: " << dy << endl;

  // return 0;

  typedef std::vector<std::vector<unsigned>> matrix;
  matrix iterations(h, std::vector<unsigned>(w));
  std::vector<uint16_t> iterations_1d(w * h);

  unsigned max_iterations = 100;

  bin_float_own bf1("1.23");
  cout << "bf1: " << bf1 << endl;
  cout << "type of bf1: " << getTypName<decltype(bf1)>() << endl;

  unsigned concurentThreadsSupported = std::thread::hardware_concurrency();
  cout << "concurentThreadsSupported: " << concurentThreadsSupported << endl;

  // TODO: add later multithreaded
  // std::thread threadObj(threadCallback,std::ref(x));
  // threadObj.join();

  for (unsigned j = 0; j < h; ++j) {
    if (j % 10 == 0) {
      cout << "j : " << j << endl;
    }

    for (unsigned i = 0; i < w; ++i) {
      bin_float_own x = x1 + dx * i;
      bin_float_own y = y1 + dy * j;

      cplx_own c(x, y);
      cplx_own z(0, 0);

      unsigned iters = 0;
      for (iters = 0; iters < max_iterations; ++iters) {
        z = z * z + c;
        // z = pow(z, const_2) + c;

        if (abs(z) > const_2) {
          break;
        }
      }

      iterations[h-1-j][i] = iters;
      iterations_1d[(h-1-j)*w+i] = iters;
    }
  }

  cout << "print iterations:" << endl;

  // FILE* file = fopen("iterations.txt", "wb");
  // std::stringstream ss("");
  // for (unsigned j = 0; j < h; ++j) {
  //   for (unsigned i = 0; i < w; ++i) {
  //     if (i > 0) {
  //       cout << ", ";
  //       ss << ",";
  //     }
  //     cout << iterations[j][i];
  //     ss << iterations[j][i];
  //   }
  //   cout << endl;
  //   ss << "\n";
  // }

  // string s = ss.str();
  // fwrite(s.c_str(), sizeof(char), s.length(), file);
  // fclose(file);

  FILE* fileBin = fopen(outputFilePath.c_str(), "wb");
  // FILE* fileBin = fopen("iterations.hex", "wb");
  uint16_t params[2] = {static_cast<uint16_t>(w), static_cast<uint16_t>(h)};
  fwrite(params, sizeof(params[0]), 2, fileBin);
  uint16_t* arr = &iterations_1d[0];
  fwrite(arr, sizeof(arr[0]), iterations_1d.size(), fileBin);
  fclose(fileBin);

  cout << "FINISHED!" << endl;

  return 0;
}
