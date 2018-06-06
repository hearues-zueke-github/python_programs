#include <stdio.h>

unsigned char l[3] = {1, 2, 3};

void g0() {
    printf("g0: outside l: %d, %d, %d\n", l[0], l[1], l[2]);
    unsigned char l[3] = {2, 3, 4};
    void g1() {
        printf("g1: outside l: %d, %d, %d\n", l[0], l[1], l[2]);
        unsigned char l[3] = {3, 4, 6};
        void g2() {
            printf("g2: outside l: %d, %d, %d\n", l[0], l[1], l[2]);
            unsigned char l[3] = {4, 5, 8};
            printf("g2:  inside l: %d, %d, %d\n", l[0], l[1], l[2]);
        }
        g2();
        printf("g1:  inside l: %d, %d, %d\n", l[0], l[1], l[2]);
    }
    g1();
    printf("g0:  inside l: %d, %d, %d\n", l[0], l[1], l[2]);
}

int main(int argc, char* argv[]) {
    g0();

    return 0;
}
