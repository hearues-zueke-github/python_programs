#include "simple_library.h"

void func_a(int foo, char bar) {
  (void) foo;
  (void) bar;
  return;
}

void func_b(char foo, int bar) {
  (void) foo;
  (void) bar;
  func_c(foo);
  return;
}

void func_c(char foo) {
  (void) foo;
  return;
}
