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


struct Tensor conv2d(struct Tensor input, struct Tensor kernel) {
    // input.shape: (N, H, W, C=IC)
    // kernel.shape: (KH, KW, OC, IC=C)
    // output.shape: (N, H, W, OC)
    int output_sz = input.shape[0] * input.shape[1] * input.shape[2] * kernel.shape[2] * 4;
    struct Tensor out;
    out.vector = (float *) malloc(output_sz);
    int shape[4] = {input.shape[0], input.shape[1], input.shape[2], kernel.shape[2]};
    memcpy(out.shape, shape, 16);
    out.sz = output_sz + 16;

    int i2 = input.shape[3];
    int i1 = input.shape[2] * i2;
    int i0 = input.shape[1] * i1;

    int k2 = kernel.shape[3];
    int k1 = kernel.shape[2] * k2;
    int k0 = kernel.shape[1] * k1;

    int o2 = kernel.shape[2];
    int o1 = input.shape[2] * o2;
    int o0 = input.shape[1] * o1;

    int odx, kdx, idx;
    for (int _n = 0; _n < input.shape[0]; _n++) {  // N
        for (int _oc = 0; _oc < kernel.shape[2]; _oc++) { // OC
            for (int _ic = 0; _ic < kernel.shape[3]; _ic++) { // IC = C
                for (int _h = 0; _h < input.shape[1]; _h++) { // H
                    for (int _w = 0; _w < input.shape[2]; _w++) { // W
                        for (int _kh = 0; _kh < kernel.shape[0]; _kh++) { // KH
                            for (int _kw = 0; _kw < kernel.shape[1]; _kw++) { // KW
                                odx = o0 * _n + o1 * _h + o2 * _w + _oc;
                                kdx = k0 * _kh + k1 * _kw + k2 * _oc + _ic;
                                idx = i0 * _n + i1 * _h + i2 * _w + _ic;
                                out.vector[odx] += kernel.vector[kdx] * input.vector[idx];
                            }
                        }
                    }
                }
            }
        }
    }
    return out;
}


int main (int argc, char* argv[]) {

    struct Tensor tensor_in = get_tensor(argv[1]);
    struct Tensor tensor_ke = get_tensor(argv[2]);

    struct Tensor tensor_ot;

    tensor_ot = conv2d(tensor_in, tensor_ke);
    write_tensor(tensor_in, "output_tensor.bin");

    free(tensor_in.vector);
    free(tensor_ke.vector);
    free(tensor_ot.vector);
    return 0;
}