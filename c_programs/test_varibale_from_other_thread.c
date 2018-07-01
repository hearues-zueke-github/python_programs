#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

pthread_t tid[2];

int* a = NULL;

void* set_variable(void* input) {
    int b = 12345;
    a = &b;
    int i;
    for (i = 0; i < 1000000; i++) {}
    
    return NULL;
}

void* get_variable(void* input) {
    printf("Value of a is: %d\n", *a);

    int i;
    for (i = 0; i < 1000000; i++) {}

    return NULL;
}

int main(int argc, char* argv[]) {
    printf("Hello World!\n");

    pthread_create(&(tid[0]), NULL, &set_variable, NULL);
    usleep(1000);
    pthread_create(&(tid[1]), NULL, &get_variable, NULL);

    void* ret_val;
    pthread_join(tid[0], &ret_val);
    pthread_join(tid[1], &ret_val);
    return 0;
}
