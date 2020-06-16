#include <stdio.h>
#include <string.h>
#include <stdlib.h>


struct Tensor {
    int shape[4];
    float* vector;
    int sz;
};


struct Tensor get_tensor(char* file_name) {
    FILE *f;
    if ((f = fopen(file_name, "rb")) == NULL) {
        printf("Error: file opening in %s", file_name);
        exit(0);
    }
    fseek(f, 0, SEEK_END);
    int file_sz = ftell(f);
    fseek(f, 0, SEEK_SET);

    int shape[4] = {-1, -1, -1, -1};
    float* vector = (float *) malloc(file_sz - 16);  // 16 for shape

    fread(shape, 4, 4, f);
    fread(vector, file_sz - 16, 1, f);

    printf("Read file: %d bytes (%d, %d, %d, %d) shape from %s\n",
            file_sz, shape[0], shape[1], shape[2], shape[3], file_name);

    struct Tensor tensor;
    tensor.vector = vector;
    memcpy(tensor.shape, shape, 16);
    tensor.sz = file_sz;
    return tensor;
}


void* write_tensor(struct Tensor tensor, char* file_name) {
    FILE *fp = fopen(file_name, "wb");
    fwrite(tensor.shape, 4, 4, fp);
    fwrite(tensor.vector, tensor.sz - 16, 1, fp);
    fclose(fp);
}


int main (int argc, char* argv[]) {

    struct Tensor tensor_in = get_tensor(argv[1]);
    struct Tensor tensor_ke = get_tensor(argv[2]);

    // write_tensor(tensor_in, "output_tensor.bin");
    free(tensor_in.vector);
    free(tensor_ke.vector);
    return 0;
}