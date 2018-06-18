#include <stdio.h>

int main(
    int argc, char* argv[]) {
    int (*f)(int a) {
        int g(int b) {
            return a+b;
        }

        return g;
    }

    return 0;
}
