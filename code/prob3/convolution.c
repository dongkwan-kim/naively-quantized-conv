#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#include <string.h>
#include <pthread.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))


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
    printf("Write tensor: %s\n", file_name);
}


struct Padding {
    int top;
    int bottom;
    int left;
    int right;
};


struct Padding get_same_padding_in_tf(int oh, int ow, int kh, int kw) {
    // strides: 1 (default)
    // padding: same (default)
    int p_vertical = max(- 1 + kh, 0);
    int p_horizontal = max(- 1 + kw, 0);
    struct Padding pad;
    pad.top = p_vertical / 2;
    pad.bottom = p_vertical - pad.top;
    pad.left = p_horizontal / 2;
    pad.right = p_horizontal - pad.left;
    return pad;
}


float* get_input_patch(struct Tensor input, struct Padding pad,
                       int _n, int _h, int _w, int kh, int kw,
                       int i0, int i1, int i2) {
    // input.shape: (N=1, H, W, C=IC)
    // --> (kh, kw, ic)
    int h = input.shape[1];
    int w = input.shape[2];
    int ic = input.shape[3];
    int patch_sz = kh * kw * ic;
    float *patch = (float *) calloc(patch_sz, sizeof(float));

    int center_h = kh / 2;
    int center_w = kw / 2;

    int input_base_idx = i0 * _n + i1 * (_h - center_h) + i2 * (_w - center_w);
    int patch_base_idx = 0;
    int size = kw * ic;
    int end_h = kh;

    if (pad.top != 0 && _h - center_h < 0) {
        input_base_idx += i1 * pad.top;
        patch_base_idx += kw * ic * pad.top;
        end_h -= pad.top;
    } else if (pad.bottom != 0 && _h - center_h + kh > h) {
        end_h -= pad.bottom;
    }
    if (pad.left != 0 && _w - center_w < 0){
        input_base_idx += i2 * pad.left;
        patch_base_idx += ic * pad.left;
        size -= ic * pad.left;
    } else if (pad.right != 0 && _w - center_w + kw > w) {
        size -= ic * pad.right;
    }
    int input_start_idx, patch_start_idx;
    for (int _kh = 0; _kh < end_h; _kh++) {
        input_start_idx = input_base_idx + i1 * _kh;
        patch_start_idx = patch_base_idx + kw * ic * _kh;
        memcpy(patch + patch_start_idx, input.vector + input_start_idx, size * sizeof(float));
    }
    return patch;
}


void dot(int dim, float* a1d, float* b1d, float* c) {

    __m256 m_sum, m_a1d, m_b1d, res;

    m_sum = _mm256_setzero_ps();

    *c = 0;

    int i;
    for (i = 0; i < dim - 7; i = i + 8) {
        m_a1d = _mm256_loadu_ps(&a1d[i]);
        m_b1d = _mm256_loadu_ps(&b1d[i]);

        res = _mm256_mul_ps(m_a1d, m_b1d);
        m_sum = _mm256_add_ps(m_sum, res);
    }
    for (; i < dim; ++i) {
        *c += a1d[i] * b1d[i];
    }
    for (int k = 0; k < 8; k++) {
        *c += m_sum[k];
    }
}


struct p_mmv_args {
    int dim_m;
    int dim_k;
    int row_idx;
    float* mxk;
    float* kx1;
    float* mx1;
} args;



void mmv(int dim_m, int dim_k, float* mxk, float* kx1, float* mx1) {
    float* row1xk = (float *) malloc(sizeof(float) * dim_k);
    int row_idx;
    for (int i = 0; i < dim_m; i ++) {
        row_idx = i * dim_k;
        memcpy(row1xk, &mxk[row_idx], dim_k * sizeof(float));
        dot(dim_k, row1xk, kx1, &mx1[i]);
    }
    free(row1xk);
}


void *p_mmv(void* arguments) {
    struct p_mmv_args *args = arguments;
    int idx = args->row_idx * args->dim_k;
    mmv(args->dim_m, args->dim_k, &args->mxk[idx], args->kx1, &args->mx1[args->row_idx]);
    free(args);
}


void mmv_pthread(int num_thread, int dim_m, int dim_k, float* mxk, float* kx1, float* mx1) {

    pthread_t tid[num_thread];

    int row_sz = dim_m / num_thread;
    struct p_mmv_args args, *_args;
    args.dim_m = row_sz;
    args.dim_k = dim_k;
    args.mxk = mxk;
    args.kx1 = kx1;
    args.mx1 = mx1;
    args.row_idx = -1;

    for (int t = 0; t < num_thread; t++) {
        _args = (struct p_mmv_args *) malloc(sizeof(struct p_mmv_args));
        memcpy(_args, &args, sizeof(struct p_mmv_args));
        _args->row_idx = t * row_sz;
        pthread_create(tid + t, NULL, p_mmv, (void *) _args);
    }

    if (dim_m % num_thread != 0) {
        _args = (struct p_mmv_args *) malloc(sizeof(struct p_mmv_args));
        memcpy(_args, &args, sizeof(struct p_mmv_args));
        _args->row_idx = num_thread * row_sz;
        _args->dim_m = dim_m - num_thread * row_sz;
        pthread_create(tid + num_thread, NULL, p_mmv, (void *) _args);
    }

    for (int t = 0; t < num_thread; t++) {
        pthread_join(tid[t], NULL);
    }
}


void einsum_hwi_hwoi_to_o(int* shape, float* v_hwi, float* v_hwoi, float* v_o, int num_thread) {
    // shape: h, w, o, i
    int h = shape[0];
    int w = shape[1];
    int o = shape[2];
    int i = shape[3];
    int hwi_idx, hwoi_idx;

    float* v_oi;
    float* v_i1;

    for (int _h = 0; _h < h; _h++) {
        for (int _w = 0; _w < w; _w++) {

            if (num_thread == 0) {

                for (int _o = 0; _o < o; _o++) {
                    for (int _i = 0; _i < i; _i++) {
                        hwi_idx = (w * i) * _h + i * _w + _i;
                        hwoi_idx = (w * o * i) * _h + (o * i) * _w + i * _o + _i;
                        v_o[_o] += v_hwi[hwi_idx] * v_hwoi[hwoi_idx];
                    }
                }

            } else {

                /* no pthread version:
                for (int _o = 0; _o < o; _o++) {
                    hwi_idx = (w * i) * _h + i * _w;
                    hwoi_idx = (w * o * i) * _h + (o * i) * _w + i * _o;
                    float tmp = 0;
                    dot(i, v_hwi + hwi_idx, v_hwoi + hwoi_idx, &tmp);
                    v_o[_o] += tmp;
                }
                */

                hwi_idx = (w * i) * _h + i * _w;
                hwoi_idx = (w * o * i) * _h + (o * i) * _w;

                v_i1 = v_hwi + hwi_idx;
                v_oi = v_hwoi + hwoi_idx;

                float *tmp_o = (float *) calloc(o, sizeof(float));
                mmv_pthread(num_thread, o, i, v_oi, v_i1, tmp_o);
                for (int _o = 0; _o < o; _o++) {
                    v_o[_o] += tmp_o[_o];
                }
            }
        }
    }
}


struct Tensor conv2d(struct Tensor input, struct Tensor kernel, int num_thread) {
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

    int o2 = kernel.shape[2];
    int o1 = input.shape[2] * o2;
    int o0 = input.shape[1] * o1;

    struct Padding pad = get_same_padding_in_tf(out.shape[1], out.shape[2], kernel.shape[0], kernel.shape[1]);

    int patch_dim = kernel.shape[0] * kernel.shape[1] * kernel.shape[3];
    float *patch_hwi;  // kh * kw * ic
    float *v_o;

    int start_odx;
    int oc = kernel.shape[2];
    for (int _n = 0; _n < input.shape[0]; _n++) {  // N
        for (int _h = 0; _h < input.shape[1]; _h++) { // H
            for (int _w = 0; _w < input.shape[2]; _w++) { // W

                start_odx = o0 * _n + o1 * _h + o2 * _w;

                v_o = (float *) calloc(oc, sizeof(float));

                patch_hwi = get_input_patch(input, pad, _n, _h, _w, kernel.shape[0], kernel.shape[1], i0, i1, i2);
                einsum_hwi_hwoi_to_o(kernel.shape, patch_hwi, kernel.vector, v_o, num_thread);

                memcpy(out.vector + start_odx, v_o, oc * sizeof(float));

                free(patch_hwi);
                free(v_o);
            }
        }
    }
    return out;
}


int main (int argc, char* argv[]) {

    clock_t start, end;
    float elapsed_time;

    struct Tensor tensor_in = get_tensor(argv[1]);
    struct Tensor tensor_ke = get_tensor(argv[2]);

    struct Tensor tensor_ot;

    int num_thread = 1;
    start = clock();
    tensor_ot = conv2d(tensor_in, tensor_ke, num_thread);
    end = clock();
    elapsed_time = (float) (end - start) / CLOCKS_PER_SEC;
    printf("Elapsed time w/ AVX+pthread: %f \n", elapsed_time);
    write_tensor(tensor_ot, "output_tensor.bin");

    free(tensor_in.vector);
    free(tensor_ke.vector);
    free(tensor_ot.vector);
    return 0;
}