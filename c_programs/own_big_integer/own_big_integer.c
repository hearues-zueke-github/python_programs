#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>

typedef struct _BigNum {
    int base;
    uint8_t* arr;
    int size;
    int last_significant_byte;
} BigNum;

typedef unsigned char uint8;

void print_big_num(BigNum bignum);

void calc_a_plus_b(BigNum a, BigNum b, BigNum** result);
void calc_a_mult_b(BigNum a, BigNum b, BigNum** result);
void calc_a_div_b(BigNum a, BigNum b, BigNum** result, BigNum** rest);

int check_a_bigger_equal_b(BigNum a, BigNum b);

void calc_a_pow_b(BigNum a, BigNum b, BigNum** result);

void create_new_big_num(BigNum** bignum) {
    BigNum* temp = malloc(sizeof(BigNum));

    temp->base = 0;
    temp->arr = NULL;
    temp->size = 0;
    temp->last_significant_byte = -1;

    *bignum = temp;
}

void delete_big_num(BigNum* bignum) {
    if (bignum->arr) {
        free(bignum->arr);
    }
    free(bignum);
}

void copy_big_num(BigNum* src, BigNum** dst) {
    BigNum* copy = malloc(sizeof(BigNum));

    copy->base = src->base;
    copy->arr = malloc(sizeof(uint8_t)*src->size);
    memcpy(copy->arr, src->arr, src->size);
    copy->size = src->size;
    copy->last_significant_byte = src->last_significant_byte;

    *dst = copy;
}

void convert_a_to_base_2(BigNum a, BigNum** result) {
    assert(a.base > 1);
    assert(a.base == 10); // only for now!

    printf("NOW: Convert to base 2!\n");

    printf("a: ");
    print_big_num(a);

    BigNum num_2;
    uint8 arr_2[1] = {2};
    num_2.base = 10;
    num_2.arr = arr_2;
    num_2.size = 1;
    num_2.last_significant_byte = 0;

    int size = 10;
    uint8* arr = malloc(sizeof(uint8)*size);

    BigNum* temp;
    copy_big_num(&a, &temp);

    printf("temp: ");
    print_big_num(*temp);

    int i;
    for (i = 0; ; i++) {
        if (i >= size) {
            size += 10;
            arr = realloc(arr, sizeof(uint8_t)*size);
        }

        BigNum* num;
        BigNum* rest;

        calc_a_div_b(*temp, num_2, &num, &rest);

        arr[i] = rest->arr[0];
        printf("temp: ");
        print_big_num(*temp);
        printf("rest: ");
        print_big_num(*rest);
        printf("arr[i]: %d\n", arr[i]);
        delete_big_num(temp);
        delete_big_num(rest);
        temp = num;

        if (temp->size == 1 && temp->arr[0] == 0) {
            delete_big_num(temp);
            break;
        }
    }

    BigNum* new_num;
    create_new_big_num(&new_num);

    new_num->base = 2;
    new_num->arr = arr;
    new_num->size = i+1;
    new_num->last_significant_byte = i;

    *result = new_num;
}

void calc_a_plus_b(BigNum a, BigNum b, BigNum** result) {
    printf("NOW: Addition, +\n");
    assert(a.base > 1);
    assert(b.base > 1);
    assert(a.base == b.base);
    
    int size_a = a.size;
    int size_b = b.size;
    int base = a.base;


    printf("a: ");
    print_big_num(a);
    printf("b: ");
    print_big_num(b);

    // get new size
    int max_size = a.size > b.size ? a.size : b.size;

    uint8_t* arr = malloc(sizeof(uint8_t)*max_size);

    int i;
    int remain = 0;
    for (i = 0; i < max_size; i++) {
        int sum = 0;
        if (i < size_a) {
            sum += a.arr[i];
        }
        if (i < size_b) {
            sum += b.arr[i];
        }
        sum += remain;

        arr[i] = sum % base;
        remain = sum / base;
    }

    if (remain > 0) {
        arr = realloc(arr, sizeof(uint8_t)*(max_size+1));
        arr[i] += remain;
        i++;
    }

    BigNum* res;
    create_new_big_num(&res);

    res->base = base;
    res->arr = arr;
    res->size = i;
    res->last_significant_byte = i-1;

    *result = res;
}

void calc_a_mult_b(BigNum a, BigNum b, BigNum** result) {
    printf("NOW: Multiplication, *\n");
    assert(a.base > 1);
    assert(b.base > 1);
    assert(a.base == b.base);

    if (a.size < b.size) {
        BigNum temp = a;
        a = b;
        b = temp;
    }

    int size_a = a.size;
    int size_b = b.size;
    int base = a.base;

    printf("a: ");
    print_big_num(a);
    printf("b: ");
    print_big_num(b);

    int max_size = size_a+size_b+1;
    uint8_t* arr = calloc(sizeof(uint8_t), max_size);

    int i;
    int j;
    for (j = 0; j < size_b; j++) {
        int num = b.arr[j];
        if (num == 0) {
            continue;
        }
        int remain = 0;
        for (i = 0; i < size_a; i++) {
            int sum = arr[j+i]+a.arr[i]*num+remain;
            arr[j+i] = sum % base;
            remain = sum / base;
        }
        if (remain > 0) {
            arr[j+i] = remain;
        }
    }

    while (arr[max_size-1] == 0) {
        max_size -= 1;
        arr = realloc(arr, sizeof(uint8_t)*max_size);
    }

    BigNum* res;
    create_new_big_num(&res);
    
    res->base = base;
    res->arr = arr;
    res->size = max_size;
    res->last_significant_byte = i-1;

    *result = res;

    printf("res: ");
    print_big_num(*res);
}

void calc_a_div_b(BigNum a, BigNum b, BigNum** result, BigNum** rest) {
    printf("NOW: Division, /\n");
    assert(a.base > 1);
    assert(b.base > 1);
    assert(a.base == b.base);

    int size_a = a.size;
    int size_b = b.size;
    int base = a.base;
    
    if (!check_a_bigger_equal_b(a, b)) {
        BigNum* res = malloc(sizeof(BigNum));
        res->base = base;
        res->arr = calloc(sizeof(uint8_t), 1);
        res->size = 1;
        res->last_significant_byte = 0;

        *result = res;

        BigNum* rest_num = malloc(sizeof(BigNum));
        rest_num->base = base;
        rest_num->arr = malloc(sizeof(uint8_t)*size_a);
        memcpy(rest_num->arr, a.arr, size_a);
        rest_num->size = size_a;
        rest_num->last_significant_byte = size_a-1;

        *rest = rest_num;

        return;
    }

    int result_length = size_a-size_b+1;
    uint8_t* result_temp = calloc(sizeof(uint8_t), result_length);
    uint8_t* rest_temp = malloc(sizeof(uint8_t)*size_a);
    memcpy(rest_temp, a.arr, size_a);

    printf("a: ");
    print_big_num(a);
    printf("b: ");
    print_big_num(b);

    BigNum temp;
    temp.base = base;
    temp.size = size_b;

    printf("result_length: %d\n", result_length);
    int i = 0;
    int k;
    int j = result_length-1;
    temp.arr = rest_temp+j;

    // First do the first digits, if possible, to substract
    while (check_a_bigger_equal_b(temp, b)) {
        result_temp[0]++;
        int remain = 0;
        for (k = 0; k < size_b; k++) {
            int n1 = temp.arr[k];
            int n2 = b.arr[k]+remain;
            if (n1 < n2) {
                temp.arr[k] = n1+base-n2;
                remain = 1;
            } else {
                temp.arr[k] = n1-n2;
                remain = 0;
            }
        }
    }
    if (result_temp[0] > 0) {
        i++;
    }

    j -= 1;
    for (; j > -1; j--) {
        int pos = j+size_b;
        temp.arr = rest_temp+j;
        // printf("pos: %d, rest_temp[pos]: %d, temp: ", pos, rest_temp[pos]);
        // print_big_num(temp);
        // Now loop over each pair of numbers until the substraction is done
        // while (check_a_bigger_equal_b(temp, b)) {
        while ((rest_temp[pos] > 0) || check_a_bigger_equal_b(temp, b)) {
            result_temp[i]++;
            int remain = 0;
            for (k = 0; k < size_b; k++) {
                int n1 = temp.arr[k];
                int n2 = b.arr[k]+remain;
                if (n1 < n2) {
                    temp.arr[k] = n1+base-n2;
                    remain = 1;
                } else {
                    temp.arr[k] = n1-n2;
                    remain = 0;
                }
            }
            if (remain > 0) {
                rest_temp[pos]--;
            }
        }
        // printf("i: %d, result_temp[i]: %d\n", i, result_temp[i]);
        i++;
    }

    temp.arr = rest_temp;
    temp.size = size_a;

    printf("temp: ");
    print_big_num(temp);

    printf("i: %d\n", i);

    if (result_length > i) {
        result_length--;
        result_temp = realloc(result_temp, sizeof(uint8_t)*result_length);
    }

    // TODO: find optimal length for rest_temp, cut of the zeŕos!

    for (i = 0, j = result_length-1; i < j; i++, j--) {
        int temp = result_temp[i];
        result_temp[i] = result_temp[j];
        result_temp[j] = temp;
    }

    BigNum* res;
    create_new_big_num(&res);
    
    res->base = base;
    res->arr = result_temp;
    res->size = result_length;
    res->last_significant_byte = result_length-1;

    *result = res;

    for (i = size_b-1; i > -1; i--) {
        if (rest_temp[i] > 0) {
            break;
        }
    }

    if (i < 0) {
        i = 0;
    }

    rest_temp = realloc(rest_temp, sizeof(uint8_t)*(i+1));

    BigNum* rest_num;
    create_new_big_num(&rest_num);
    
    rest_num->base = base;
    rest_num->arr = rest_temp;
    rest_num->size = i+1;
    rest_num->last_significant_byte = i;

    *rest = rest_num;
}

int check_a_bigger_equal_b(BigNum a, BigNum b) {
    assert(a.base > 1);
    assert(b.base > 1);
    assert(a.base == b.base);

    if (a.size > b.size) {
        return 1;
    } else if (a.size < b.size) {
        return 0;
    }

    int i;
    int size = a.size;
    uint8_t* arr_a = a.arr;
    uint8_t* arr_b = b.arr;
    for (i = size-1; i > -1; i--) {
        if (arr_a[i] > arr_b[i]) {
            return 1;
        } else if (arr_a[i] < arr_b[i]) {
            return 0;
        }
    }

    return 1; // in this case a and b are equal!
}

void calc_a_pow_b(BigNum a, BigNum b, BigNum** result) {
    printf("NOW: Pow Pow Pow, /\n");
    assert(a.base > 1);
    assert(b.base > 1);
    assert(a.base == b.base);

    printf("a: ");
    print_big_num(a);
    printf("b: ");
    print_big_num(b);

    BigNum* temp;
    copy_big_num(&a, &temp);
    BigNum* p;
    create_new_big_num(&p);
    p->arr = malloc(sizeof(uint8_t)*1);
    p->base = a.base;
    p->arr[0] = 1;
    p->size = 1;
    p->last_significant_byte = 0;

    BigNum* b_base2;
    convert_a_to_base_2(b, &b_base2);

    int size = b_base2->size-1;
    int i;
    for (i = 0; ; i++) {
        if (1 || b_base2->arr[i]) {
            BigNum* temp3;
            printf("temp: %p\n", temp);
            printf("p: %p\n", p);
            calc_a_mult_b(*temp, *p, &temp3);
            delete_big_num(p);
            p = temp3;
        }

        if (i >= size) {
            break;
        }

        BigNum* temp2;
        printf("temp: %p\n", temp);
        calc_a_mult_b(*temp, *temp, &temp2);
        // delete_big_num(temp);
        // temp = temp2;
    }

    delete_big_num(b_base2);
    delete_big_num(temp);

    *result = p;
}

void print_big_num(BigNum bignum) {
// void print_big_num(int base, uint8_t* arr, int size) {
    int base = bignum.base;
    uint8_t* arr = bignum.arr;
    int size = bignum.size;
    // int last_significant_byte = bignum.last_significant_byte;

    int num_size = (base <= 10 ? 1 : base <= 100 ? 2 : 3);
    char* buffer = malloc(sizeof(char)*((num_size+2)*size));
    char format[6] = {0};
    sprintf(format, "%%%dd, ", num_size);
    int i = 0;
    for (i = 0; i < size; i++) {
        sprintf(buffer+i*(num_size+2), format, arr[i]);
    }
    // sprintf(buffer+size*(num_size+2)-2, "\0");
    *(buffer+size*(num_size+2)-2) = 0;

    printf("%s; size: %d\n", buffer, bignum.size);

    free(buffer);
}

int main(int argc, char* argv[]) {
    struct timeval time; 
    gettimeofday(&time,NULL);
    srand((time.tv_sec * 1000) + (time.tv_usec / 1000));
    
    int base = 10;
    
    BigNum bignum_a;
    bignum_a.base = base;
    uint8_t arr_a[3] = {2, 3, 9};
    bignum_a.arr = arr_a;
    bignum_a.size = 3;
    bignum_a.last_significant_byte = 2;

    BigNum bignum_b;
    bignum_b.base = base;
    uint8_t arr_b[4] = {8, 5, 2, 4};
    bignum_b.arr = arr_b;
    bignum_b.size = 4;
    bignum_b.last_significant_byte = 3;

    BigNum* bignum_result;
    BigNum* bignum_result_mult;
    BigNum* bignum_result_div;
    BigNum* bignum_result_div_rest;
    BigNum* bignum_result_base2;
    BigNum* bignum_result_pow;


    calc_a_plus_b(bignum_a, bignum_b, &bignum_result);

    printf("result: ");
    print_big_num(*bignum_result);


    calc_a_mult_b(bignum_a, bignum_b, &bignum_result_mult);

    printf("result: ");
    print_big_num(*bignum_result_mult);


    calc_a_div_b(bignum_a, bignum_b, &bignum_result_div, &bignum_result_div_rest);

    printf("result: \n");
    print_big_num(*bignum_result_div);

    printf("result_rest: \n");
    print_big_num(*bignum_result_div_rest);

    convert_a_to_base_2(bignum_a, &bignum_result_base2);

    printf("result_base_2: \n");
    print_big_num(*bignum_result_base2);

    calc_a_pow_b(bignum_a, bignum_b, &bignum_result_pow);

    printf("result_pow: \n");
    print_big_num(*bignum_result_pow);


    delete_big_num(bignum_result);
    delete_big_num(bignum_result_mult);
    delete_big_num(bignum_result_div);
    delete_big_num(bignum_result_div_rest);
    delete_big_num(bignum_result_base2);

    return 0;
}
