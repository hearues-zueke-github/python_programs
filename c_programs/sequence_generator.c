#include <stdio.h>
#include <stdlib.h>
// #include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <limits.h>

void print_arr (uint8_t* arr, size_t size, char* arr_name) {
  printf("%s:\n", arr_name);
  printf("%2d", arr[0]);
  int i = 0;
  for (i = 1; i < size; i++) {
  printf(", %2d", arr[i]);
  }
  printf("\n");
}

int main (int argc, char* argv[]) {
  size_t n = 10000;
  uint8_t* arr = malloc(n);

  if (arr == NULL) {
  printf("Memory Management Error!\n");
  return -1;
  }

  // set the arr beginning!
  arr[0] = 1;

  int m = 10;
  int i = 0;
  for (i = 0; i < n - 1; i++) {
    int s = 0;
    int j = i;
    int acc = 0;
    int multiplier = 1;
    while (j >= 0) {
      int x = arr[j];
      s = (s + x * multiplier) % m;
      multiplier = (multiplier + 1) % m;
      acc += x + 1;
      j -= acc;
    }
    arr[i+1] = s;
  }

  // print_arr(arr, n, "arr");
  printf("Finished calculating the sequence!\n");

  char full_path[PATH_MAX];
  char cwd[PATH_MAX];
  if (getcwd(cwd, sizeof(cwd)) != NULL) {
    printf("Current working dir: %s\n", cwd);
  } else {
    perror("getcwd() error");
    return 1;
  }

  sprintf(full_path, "%s/../sequence_generators/sequence_arr.hex", cwd);

  printf("cwd: %s\n", cwd);
  printf("full_path: %s\n", full_path);

  FILE* file_out = fopen(full_path, "wb");
  // FILE* file_out = fopen("sequence_arr.hex", "wb");
  
  fwrite(arr, n, sizeof(arr[0]), file_out);

  fclose(file_out);

  free(arr);

  return 0;
}
