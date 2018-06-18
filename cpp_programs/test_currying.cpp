#include <iostream>

int main(int argc, char* argv[]) {
  auto f = [](int a) {
    auto g = [a](int b) {
      return a+b;
    };
    return g;
  };

  auto g = f(4);

  std::cout << "g(3): " << g(3) << std::endl;

  return 0;
}
