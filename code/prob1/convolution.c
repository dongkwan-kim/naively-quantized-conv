#include <stdio.h>
#include <string.h>
#include <stdlib.h>


float* get_tensor(char* file_name) {
    FILE *f;
    if ((f = fopen(file_name, "rb")) == NULL) {
        printf("Error: file opening in %s", file_name);
        exit(0);
    }
    fseek(f, 0, SEEK_END);
    int file_sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    float* tensor = (float *) malloc(file_sz);
    fread(tensor, file_sz, 1, f);
    printf("Read file: %d of %s\n", file_sz, file_name);
    return tensor;
}

int main (int argc, char* argv[]) {

    float* tensor_in = get_tensor(argv[1]);
    float* tensor_ke = get_tensor(argv[2]);

    free(tensor_in);
    free(tensor_ke);
    return 0;
}