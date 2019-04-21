#include <stdio.h>
#include <stdlib.h>

void print_array(int* arr, int length) {
  int i;
  printf("[");
  for (i = 0; i < length; i++) {
    if (i > 0) {
      printf(", ");
    }
    printf("%d", arr[i]);
  }
  printf("]");
  printf("\n");
}

void delete_max(int** arr, int* length) {
  /* TODO: write your code! */
  int i;

  int max_val = 0;
  int max_idx = 0;

  for (i = 0; i < *length; i++) {
    int val = (*arr)[i];

    if (max_val < val) {
      max_val = val;
      max_idx = i;
    }
  }

  printf("max_val: %d\n", max_val);
  printf("max_idx: %d\n", max_idx);

  for (i = max_idx; i < *length-1; i++) {
    (*arr)[i] = (*arr)[i+1];
  }

  *arr = realloc(*arr, sizeof(int) * (*length-1));
  (*length)--;
}

void mirror_array(int* arr, int length) {
  /* TODO: write your code! */
  int len = length / 2;
  printf("len: %d\n", len);

  int i;
  for (i = 0; i < len; i++) {
    int temp = arr[i];
    arr[i] = arr[length-1-i];
    arr[length-1-i] = temp;
  }
}

int concat_arrays(int** arr, int* len, int* arr1, int len1, int* arr2, int len2) {
  /* TODO: write your code! */
  *len = len1 + len2;
  *arr = malloc(*len * sizeof(int));

  if (*arr == NULL) {
    *len = 0;
    return -1;
  }

  int i;
  for (i = 0; i < len1; i++) {
    (*arr)[i] = arr1[i];
  }
  for (i = 0; i < len2; i++) {
    (*arr)[i+len1] = arr2[i];
  }

  return 0;
}

int main(int argc, char* argv[]) {
  /* TODO: write your code! */
  int len1 = 5;
  int* arr1 = malloc(len1 * sizeof(int));

  if (arr1 == NULL) {
    printf("NO MEMORY!\n");
    return -1;
  }

  arr1[0] = 2;
  arr1[1] = 1;
  arr1[2] = 8;
  arr1[3] = 4;
  arr1[4] = 6;

  int len2 = 4;
  int* arr2 = malloc(len1 * sizeof(int));

  if (arr2 == NULL) {
    free(arr1);
    printf("NO MEMORY!\n");
    return -1;
  }

  arr2[0] = 21;
  arr2[1] = 11;
  arr2[2] = 81;
  arr2[3] = 41;


  printf("arr1:\n");
  print_array(arr1, len1);

  int excersice_num = 2;

  switch (excersice_num) {
    case 0:
      delete_max(&arr1, &len1);
      printf("arr1 after delete max:\n");
      print_array(arr1, len1);
      break;
    
    case 1:
      mirror_array(arr1, len1);
      printf("arr1 after mirror:\n");
      print_array(arr1, len1);
      break;

    case 2:
      printf("arr2:\n");
      print_array(arr2, len2);

      int* arr_con = NULL;
      int len_con = 0;
      int ret = concat_arrays(&arr_con, &len_con, arr1, len1, arr2, len2);
      if (ret) {
        free(arr1);
        free(arr2);
        printf("NO MEMORY!\n");
        return -2;
      }

      printf("after:\nlen_con: %d\n", len_con);
      printf("arr_con:\n");
      print_array(arr_con, len_con);
      break;
  }

  return 0;
}
