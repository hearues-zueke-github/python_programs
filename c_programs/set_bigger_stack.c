#include <stdio.h>

void f(int a) {
    printf("a: %d\n", a);
    f(a+1);
}

int main(int argc, char* argv[]) {
    f(0);
    
    return 0;
}
