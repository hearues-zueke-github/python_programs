#include "simple_library.h"

void func_a(int arg1, char arg2, void* arg3) {
  (void) arg1;
  (void) arg2;
  (void) arg3;

  return;
}

// in func_b the func_c is called
void func_b(char arg1, int arg2) {
  (void) arg1;
  (void) arg2;

  func_c(0);

  return;
}

void func_c(char arg1) {
  (void) arg1;

  return;
}
