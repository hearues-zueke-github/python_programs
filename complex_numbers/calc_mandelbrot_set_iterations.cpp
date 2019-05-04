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

string outputPrefix;

template <class T>
string getTypName ()
{
  typedef typename std::remove_reference<T>::type TR;
  std::unique_ptr<char, void(*)(void*)> own (
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
typedef number<cpp_bin_float<30>> bin_float_own;
typedef cpp_complex<30> cplx_own;

const bin_float_own CONST_2("2");

template <typename T> std::string type_name();

void testingBoostFloatComplex () {
  cplx_own c1(2, 3);
  cplx_own c2(3, 7);

  cout << outputPrefix << "c1: " << c1 << endl;
  cout << outputPrefix << "c2: " << c2 << endl;
  
  cplx_own c_plus = c1+c2;
  cout << outputPrefix << "c_plus: " << c_plus << endl;
  cplx_own c_minus = c1-c2;
  cout << outputPrefix << "c_minus: " << c_minus << endl;
  cplx_own c_mul = c1*c2;
  cout << outputPrefix << "c_mul: " << c_mul << endl;
  cplx_own c_div = c1/c2;
  cout << outputPrefix << "c_div: " << c_div << endl;

  cout << outputPrefix << "real_part of c1: " << real(c1)/7 << endl;
  cout << outputPrefix << "imag_part of c1: " << imag(c1)/7 << endl;

  // for getting the type_name of a type as string
  cout << outputPrefix << "typename of cplx_own: " << typeid(cplx_own::value_type).name() << endl;
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

class Index {
public:
  unsigned x;
  unsigned y;
  cplx_own c;
  cplx_own z;

  Index (unsigned x, unsigned y, cplx_own c, cplx_own z) : x(x), y(y), c(c), z(z) {

  }

  Index (const Index& old_obj) : x(old_obj.x), y(old_obj.y), c(old_obj.c), z(old_obj.z) {

  }
};

void doIterationsBegin (const unsigned h, const unsigned w, const bin_float_own y1, const bin_float_own x1, const bin_float_own dy, const bin_float_own dx, const unsigned max_iterations, std::vector<Index>& restIndices, std::vector<uint16_t>& iterations_1d) {
  for (unsigned j = 0; j < h; ++j) {
    // if (j % 10 == 0) {
    //   cout << outputPrefix << "j : " << j << endl;
    // }

    for (unsigned i = 0; i < w; ++i) {
      bin_float_own x = x1 + dx * i;
      bin_float_own y = y1 + dy * j;

      cplx_own c(x, y);
      cplx_own z(0, 0);

      unsigned iters = 0;
      for (iters = 0; iters < max_iterations; ++iters) {
        z = z * z + c;
        if (abs(z) > CONST_2) {
          break;
        }
      }

      if (iters == max_iterations) {
        restIndices.push_back(Index(i, j, c, z));
        // matCplx[j][i] = z;
      } else {
        iterations_1d[(h-1-j)*w+i] = iters;
      }
    }
  }
}

void doIterations (const unsigned h, const unsigned w, const unsigned max_iterations, const std::vector<Index> restIndices, std::vector<Index>& restIndicesNew, std::vector<uint16_t>& iterations_1d) {
  unsigned size = restIndices.size();
  unsigned size1_10th = size/10;
  for (unsigned i = 0; i < size; ++i) {
    // if (i % size1_10th == 0) {
    //   cout << outputPrefix << "doIterations: i: " << i << endl;
    // }

    Index idx(restIndices[i]);

    cplx_own c(idx.c);
    cplx_own z(idx.z);

    unsigned iters = 0;
    for (iters = 0; iters < max_iterations; ++iters) {
      z = z * z + c;
      if (abs(z) > CONST_2) {
        break;
      }
    }

    if (iters == max_iterations) {
      restIndicesNew.push_back(Index(idx.x, idx.y, c, z));
    } else {
      iterations_1d[(h-1-idx.y)*w+idx.x] = iters;
    }    
  }
}

void doMandelbrotsetIterations(const std::vector<string>& ps) {
  unsigned h = std::stoi(ps[0]); // 0x12;
  unsigned w = std::stoi(ps[1]); // 0x18;
  bin_float_own y0(ps[2]);
  bin_float_own x0(ps[3]);
  bin_float_own w1(ps[4]);
  unsigned max_iterations = std::stoi(ps[5]);
  string outputFilePath = ps[6];

  bin_float_own h1 = bin_float_own(h) / bin_float_own(w) * w1;

  bin_float_own x1 = x0 - w1 / CONST_2;
  bin_float_own y1 = y0 - h1 / CONST_2;

  bin_float_own x2 = x0 + w1 / CONST_2;
  bin_float_own y2 = y0 + h1 / CONST_2;

  bin_float_own dx = (x2 - x1) / w;
  bin_float_own dy = (y2 - y1) / h;

  // cout << outputPrefix << "w: " << w << endl;
  // cout << outputPrefix << "h: " << h << endl;

  // cout << outputPrefix << "w1: " << w1 << endl;
  // cout << outputPrefix << "h1: " << h1 << endl;

  // cout << outputPrefix << "x0: " << x0 << endl;
  // cout << outputPrefix << "y0: " << y0 << endl;

  // cout << outputPrefix << "x1: " << x1 << endl;
  // cout << outputPrefix << "y1: " << y1 << endl;

  // cout << outputPrefix << "x2: " << x2 << endl;
  // cout << outputPrefix << "y2: " << y2 << endl;

  // cout << outputPrefix << "dx: " << dx << endl;
  // cout << outputPrefix << "dy: " << dy << endl;

  // typedef std::vector<std::vector<unsigned>> matrix;
  // matrix iterations(h, std::vector<unsigned>(w));
  
  // typedef std::vector<std::vector<cplx_own>> matrixComplex;
  // matrixComplex matCplx(h, std::vector<cplx_own>(w));

  std::vector<Index> restIndices;
  std::vector<uint16_t> iterations_1d(w * h);

  // unsigned max_iterations = 100;

  // bin_float_own bf1("1.23");
  // cout << outputPrefix << "bf1: " << bf1 << endl;
  // cout << outputPrefix << "type of bf1: " << getTypName<decltype(bf1)>() << endl;

  // unsigned concurentThreadsSupported = std::thread::hardware_concurrency();
  // cout << outputPrefix << "concurentThreadsSupported: " << concurentThreadsSupported << endl;

  // TODO: add later multithreaded (maybe, maybe not)
  // std::thread threadObj(threadCallback,std::ref(x));
  // threadObj.join();

  // TODO: do an increment approach for each point, e.g. do every 100 increment of the current_max_iterations (starting with 100 e.g.)
  // TODO: next, if a point reach the current_max_iterations (100, 200, 300, etc.), save the value in a vector, and go to next point (in y and x axis)
  //       in this case it is j and i iterators. repeat this until the ultimate max_iters is reached, which is the max_iterations

  unsigned current_max_iterations = 100;

  doIterationsBegin(h, w, y1, x1, dy, dx, current_max_iterations, restIndices, iterations_1d);
  // cout << outputPrefix << "restIndices.size(): " << restIndices.size() << endl;
  unsigned oldSize = restIndices.size();

  unsigned min_diff = ((double)(h * w) * 0.008);
  // cout << outputPrefix << "min_diff: " << min_diff << endl;

  for (unsigned i = 0; i < 10; ++i) {
    // cout << outputPrefix << "Doing loop for 'doIterations'! i: " << i << endl;
    std::vector<Index> restIndicesNew;
    doIterations(h, w, current_max_iterations, restIndices, restIndicesNew, iterations_1d);
    restIndices = restIndicesNew;
    restIndicesNew.clear();
    // cout << outputPrefix << "restIndices.size(): " << restIndices.size() << endl;
    unsigned newSize = restIndices.size();

    cout << outputPrefix << "oldSize: " << oldSize << ", newSize: " << newSize << endl;

    if (oldSize < newSize + min_diff) {
      break;
    }

    oldSize = newSize;
  }

  
  // // std::vector<Index> restIndicesNew2;
  // doIterations(h, w, current_max_iterations, restIndices, restIndicesNew, iterations_1d);
  // restIndices = restIndicesNew;
  // restIndicesNew.clear();
  // cout << outputPrefix << "restIndices.size(): " << restIndices.size() << endl;

  // cout << outputPrefix << "restIndicesNew.size(): " << restIndicesNew.size() << endl;
  // cout << outputPrefix << "restIndicesNew2.size(): " << restIndicesNew2.size() << endl;
  

  // Index idx = restIndices[0];
  // cout << outputPrefix << "Index: " << "x: " << idx.x << ", y: " << idx.y << ", c: " << idx.c << ", z: " << idx.z << endl;

  // cout << outputPrefix << "print iterations:" << endl;

  FILE* fileBin = fopen(outputFilePath.c_str(), "wb");

  uint16_t params[3] = {static_cast<uint16_t>(max_iterations), static_cast<uint16_t>(h), static_cast<uint16_t>(w)};
  fwrite(params, sizeof(params[0]), 3, fileBin);
  uint16_t* arr = &iterations_1d[0];
  fwrite(arr, sizeof(arr[0]), iterations_1d.size(), fileBin);

  fclose(fileBin);
}

int main (int argc, char* argv[]) {
  cout << outputPrefix << std::setprecision(std::numeric_limits<typename cplx_own::value_type>::digits10);

  assert (argc >= 2);

  string filePath = argv[1];
  bool exists = fileExists(filePath);
  cout << outputPrefix << "File '" << filePath << "'exists: " << exists << endl;
  assert(exists);

  if (argc >= 3) {
    outputPrefix = argv[2];
  }

  // Read whole content into file f!
  std::ifstream f(filePath, std::ios::binary);
  std::vector<char> vecParams(std::istreambuf_iterator<char>{f}, {});
  cout << outputPrefix << "vecParams.size(): " << vecParams.size() << endl;

  string paramsFile = &vecParams[0];
  cout << outputPrefix << "paramsFile: " << paramsFile << endl;

  std::vector<string> lines;
  boost::split(lines, paramsFile, boost::is_any_of("\n"));
  
  for (unsigned lineNr = 0; lineNr < lines.size(); ++lineNr) {
    std::vector<string> ps;
    boost::split(ps, lines[lineNr], boost::is_any_of(","));
    for (unsigned i = 0; i < ps.size(); ++i) {
      cout << outputPrefix << "i: " << i << ", ps["<< i << "]: " << ps[i] << endl;
    }

    if (ps.size() != 7) {
      cout << outputPrefix << "Jump too next!" << endl;
      continue;
    }

    doMandelbrotsetIterations(ps);
    // TODO: create the finished pix arr here
    // TODO: finish
  }

  cout << outputPrefix << "FINISHED!" << endl;

  return 0;
}
