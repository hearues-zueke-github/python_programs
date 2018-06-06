#include <stdio.h>

unsigned char l[3] = {1, 2, 3};

void g() {
    printf("outside g: l: %d, %d, %d\n", l[0], l[1], l[2]);
    void g1() {
        printf("outside g1: l: %d, %d, %d\n", l[0], l[1], l[2]);
        unsigned char l[3] = {2, 3, 4};
        void g2() {
            printf("outside g2: l: %d, %d, %d\n", l[0], l[1], l[2]);
            unsigned char l[3] = {3, 4, 5};
            printf("inside g2: l: %d, %d, %d\n", l[0], l[1], l[2]);
        }
        g2();
        printf("inside g1: l: %d, %d, %d\n", l[0], l[1], l[2]);
    }
    g1();
    printf("inside g: l: %d, %d, %d\n", l[0], l[1], l[2]);
}

int main(int argc, char* argv[]) {
    g();

    return 0;
}
