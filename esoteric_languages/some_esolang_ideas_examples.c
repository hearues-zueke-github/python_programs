// a small C example:
bool check_if_prime(int num) {
    int i = 2;
    while (i < num) {
        if ((num % i) == 0) {
            return false;
        }
        i += 1;
    }
    return true;
}

int next_prime(int num) {
    while (!check_if_prime(num)) {
        num += 1;
    }
    return num;
}

int num = 56;
int prime = next_prime(num);

// more modified:
void check_if_prime(bool* ret, int* num) {
    int i;
    i = 2;
    while (i < (*num)) {
        if (((*num) % i) == 0) {
            ret = false;
            return;
        }
        i = i + 1;
    }
    (*ret) = true;
}

void next_prime(int* ret, int* num) {
    bool is_prime;
    check_if_prime(&is_prime, num);
    while (!is_prime) {
        (*num) = (*num) + 1;
        check_if_prime(&is_prime, num);
    }
    (*ret) = (*num);
}

int num;
num = 56;
int prime;
next_prime(&prime, &num);

// one more step:
void check_if_prime(uint8_t* ret, uint8_t* num) {
    uint8_t i[4];
    *(int*)i = 2;
    while (*(int*)i < (*(int*)num)) {
        if (((*(int*)num) % *(int*)i) == 0) {
            ret = false;
            return;
        }
        *(int*)i = *(int*)i + 1;
    }
    *(int*)ret = true;
}

void next_prime(uint8_t* ret, uint8_t* num) {
    uint8_t is_prime[4];
    check_if_prime(&is_prime, num);
    while (!is_prime) {
        (*num) = (*num) + 1;
        check_if_prime(&is_prime, num);
    }
    (*ret) = (*num);
}

uint8_t num[4];
*(int*)num = 56;
uint8_t prime[4];
next_prime(prime, num);

// my esolang style C code:
void f1(uint8_t* v1, uint8_t* v2) {
    uint8_t i[4];
    *(int*)i = 2;
    while (*(int*)i < (*(int*)v2)) {
        if (((*(int*)v2) % *(int*)i) == 0) {
            v1 = false;
            return;
        }
        *(int*)i = *(int*)i + 1;
    }
    *(int*)v1 = true;
}

void f2(uint8_t* v1, uint8_t* v2) {
    uint8_t v3[4];
    f1(&v3, v2);
    while (!v3) {
        (*v2) = (*v2) + 1;
        f1(&v3, v2);
    }
    (*v1) = (*v2);
}

uint8_t v1[4];
*(int*)v1 = 56;
uint8_t v2[4];
f2(v2, v1);

// and converted into the esolang code:
df1v1v2dv3m4gv3m4n2w<v3m4v2m4eyi=%v2m4v3m4n0eygv1n0reee
void f1(uint8_t* v1, uint8_t* v2) {
    uint8_t v3[4];
    *(int*)v3 = 2;
    while (*(int*)v3 < (*(int*)v2)) {
        if (((*(int*)v2) % *(int*)v3) == 0) {
            v1 = false;
            return;
        }
        *(int*)v3 = *(int*)v3 + 1;
    }
    *(int*)v1 = true;
}

void f2(uint8_t* v1, uint8_t* v2) {
    uint8_t v3[4];
    f1(&v3, v2);
    while (!v3) {
        (*v2) = (*v2) + 1;
        f1(&v3, v2);
    }
    (*v1) = (*v2);
}

uint8_t v1[4];
*(int*)v1 = 56;
uint8_t v2[4];
f2(v2, v1);

dv3m4a2 <-> dv3a8
int v3[2];
uint8_t v3[8];
