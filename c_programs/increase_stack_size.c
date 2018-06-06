#include <stdio.h>
#include <sys/resource.h>

void f(int a) {
    printf("a: %d\n", a);
    f(a+1);
}

void foo(void);

int main(int argc, char *argv[]) {
  struct rlimit lim = {1000000, 1000000};

  printf("limiting stack size\n");
  if (setrlimit(RLIMIT_STACK, &lim) == -1) {
    printf("rlimit failed\n");
    return 1;
  }

  foo();
  // f(0);

  return 0;
}

void foo() {
  unsigned ints[1000000];

  printf("foo: %u\n", ints[9999]=42);
}
