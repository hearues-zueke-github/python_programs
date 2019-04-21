#include <stdio.h>

int countValue(int* arr, int len, int val)
{
    int count_val = 0;
    int idx = 0;
    
    for (idx = 0; idx < len; idx++)
    {
        if (val == arr[idx])
        {
            count_val++;
        }
    }

    return count_val;
}

void f1()
{
    char str1_orig[] = "ABCDEF";
    char str2_orig[100] = {0};

    char* str1 = str1_orig;
    char* str2 = str2_orig;

    printf("before:");
    printf("str1: %p\n", str1);
    printf("str2: %p\n", str2);

    // while (str1 != '\0')
    // {
    //    str2++ = str1++;
    // }

    while (*str1 != '\0')
    {
       *(str2++) = *(str1++);
    }

    // while (str1[0] != '\0')
    // {
    //    (str2++)[0] = (str1++)[0];
    // }

    printf("after:");
    printf("str1: %p\n", str1);
    printf("str2: %p\n", str2);

    printf("string str1_orig: %s\n", str1_orig);
    printf("string str2_orig: %s\n", str2_orig);
}

int minVal(int a, int b, int c) // (2, 2, 1)
{
    if (a <= b && a <= c)
    {
        return a;
    }
    else if (b <= c && b <= a)
    {
        return b;
    }
    else
    {
        return c;
    }

    // int temp = a;
    // if (temp > b)
    // {
    //     temp = b;
    // }
    // if (temp > c)
    // {
    //     temp = c;
    // }
    // return temp
}


// char* copyInitial(char* src)
// {
//     // bla bla bla
//     for (int counter = 0; counter < 3; counter++)
//     {
//         new_str[counter] = src[counter];
//     }
//     return new_str;
// }

void f2()
{
    int number = 20;
    int i = 0;
    printf("number: %d\n", number);
    do
    {
        int number = number / 2;
        printf("%d %d\n", ++i, number);
        // exit(-2);
    } while (number > 10);
}

int main()
{
    printf("Hello World!\n");

    double a = (3+4+5) / 3.;

    printf("a: %f\n", a);

    int arr[] = {1,2,3,2,3,3,4,5,6};
    int len = 9;

    int count = countValue(arr, len, 3);
    printf("count: %d\n", count);

    char* str = "abcde";

    int len2 = 0;
    while (str[++len2]) {}
    printf("len2: %d\n", len2);

    // f1();
    f2();

    return 0;
}

