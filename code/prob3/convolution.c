#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>
#include <pthread.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))


float get_nrmse(float* xs, float* ys, int dim) {
    float y_max = ys[0];
    float y_min = ys[0];
    float x, y;
    float accum = 0;
    for (int i = 0; i < dim; i++) {
        x = xs[i];
        y = ys[i];

        accum += pow(x - y, 2);

        if (y >= y_max)
            y_max = y;
        if (y <= y_min)
            y_min = y;

    }
    accum = accum / ((float) dim);
    accum = sqrtf(accum) / (y_max - y_min);
    return accum;
}


struct Tensor {
    int shape[4];
    float* vector;
    int32_t* v32;
    int16_t* v16;
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

struct Tensor quantize(struct Tensor t, int quant_byte, float quant_const) {
    int dim = t.shape[0] * t.shape[1] * t.shape[2] * t.shape[3];
    if (quant_byte == 32) {
        int32_t* qv = (int32_t *) malloc(dim * sizeof(int32_t));
        for (int i = 0; i < dim; i++)
            qv[i] = (int32_t) (t.vector[i] * quant_const);
        t.v32 = qv;
    } else if (quant_byte == 16) {
        int16_t* qv = (int16_t *) malloc(dim * sizeof(int16_t));
        for (int i = 0; i < dim; i++)
            qv[i] = (int16_t) (t.vector[i] * quant_const);
        t.v16 = qv;
    }
    return t;
}

struct Tensor recover_tensor(struct Tensor tensor, int quant_byte, float quant_const) {

    if (quant_byte <= 0) {
        return tensor;
    }

    int dim = tensor.shape[0] * tensor.shape[1] * tensor.shape[2] * tensor.shape[3];
    float quant_const_square = quant_const * quant_const;

    float *vector = (float *) malloc(dim * sizeof(float));
    tensor.vector = vector;

    if (quant_byte == 32) {
        for (int i = 0; i < dim; i++)
            tensor.vector[i] = tensor.v32[i] / quant_const_square;
    } else if (quant_byte == 16) {
        for (int i = 0; i < dim; i++)
            tensor.vector[i] = tensor.v16[i] / quant_const_square;
    }
    tensor.sz = dim * sizeof(float);
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

int32_t* get_input_patch_32(struct Tensor input, struct Padding pad,
                            int _n, int _h, int _w, int kh, int kw,
                            int i0, int i1, int i2) {
    // input.shape: (N=1, H, W, C=IC)
    // --> (kh, kw, ic)
    int h = input.shape[1];
    int w = input.shape[2];
    int ic = input.shape[3];
    int patch_sz = kh * kw * ic;
    int32_t *patch = (int32_t *) calloc(patch_sz, sizeof(int32_t));

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
        memcpy(patch + patch_start_idx, input.v32 + input_start_idx, size * sizeof(int32_t));
    }
    return patch;
}

int16_t* get_input_patch_16(struct Tensor input, struct Padding pad,
                            int _n, int _h, int _w, int kh, int kw,
                            int i0, int i1, int i2) {
    // input.shape: (N=1, H, W, C=IC)
    // --> (kh, kw, ic)
    int h = input.shape[1];
    int w = input.shape[2];
    int ic = input.shape[3];
    int patch_sz = kh * kw * ic;
    int16_t *patch = (int16_t *) calloc(patch_sz, sizeof(int16_t));

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
        memcpy(patch + patch_start_idx, input.v16 + input_start_idx, size * sizeof(int16_t));
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


void dot_32(int dim, int32_t* a1d, int32_t* b1d, int32_t* c) {

    __m256i m_sum, m_a1d, m_b1d, res;

    m_sum = _mm256_setzero_si256();

    *c = 0;

    int i;
    for (i = 0; i < dim - 7; i = i + 8) {
        m_a1d = _mm256_loadu_si256((__m256i*) &a1d[i]);
        m_b1d = _mm256_loadu_si256((__m256i*) &b1d[i]);

        res = _mm256_mullo_epi32(m_a1d, m_b1d);
        m_sum = _mm256_add_epi32(m_sum, res);
    }

    for (; i < dim; ++i) {
        *c += a1d[i] * b1d[i];
    }

    int32_t* m_sum_32 = (int32_t *)&m_sum;
    for (int k = 0; k < 8; k++) {
        *c += m_sum_32[k];
    }
}

void dot_16(int dim, int16_t* a1d, int16_t* b1d, int16_t* c) {

    int16_t ans = 0;
    for (int i = 0; i < dim; i++)
        ans += a1d[i] * b1d[i];

    __m256i m_sum, m_a1d, m_b1d, res;

    m_sum = _mm256_setzero_si256();

    *c = 0;

    int i;
    for (i = 0; i < dim - 15; i = i + 16) {
        m_a1d = _mm256_loadu_si256((__m256i*) &a1d[i]);
        m_b1d = _mm256_loadu_si256((__m256i*) &b1d[i]);

        res = _mm256_mullo_epi16(m_a1d, m_b1d);
        m_sum = _mm256_add_epi16(m_sum, res);
    }

    for (; i < dim; ++i) {
        *c += a1d[i] * b1d[i];
    }

    int16_t* m_sum_16 = (int16_t *)&m_sum;
    for (int k = 0; k < 16; k++) {
        *c += m_sum_16[k];
    }
}


struct p_mmv_args {
    int dim_m;
    int dim_k;
    int row_idx;
    float* mxk;
    float* kx1;
    float* mx1;
    int32_t* mxk_32;
    int32_t* kx1_32;
    int32_t* mx1_32;
    int16_t* mxk_16;
    int16_t* kx1_16;
    int16_t* mx1_16;
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

void mmv_32(int dim_m, int dim_k, int32_t* mxk, int32_t* kx1, int32_t* mx1) {
    int32_t* row1xk = (int32_t *) malloc(sizeof(int32_t) * dim_k);
    int row_idx;
    for (int i = 0; i < dim_m; i ++) {
        row_idx = i * dim_k;
        memcpy(row1xk, &mxk[row_idx], dim_k * sizeof(int32_t));
        dot_32(dim_k, row1xk, kx1, &mx1[i]);
    }
    free(row1xk);
}

void mmv_16(int dim_m, int dim_k, int16_t* mxk, int16_t* kx1, int16_t* mx1) {
    int16_t* row1xk = (int16_t *) malloc(sizeof(int16_t) * dim_k);
    int row_idx;
    for (int i = 0; i < dim_m; i ++) {
        row_idx = i * dim_k;
        memcpy(row1xk, &mxk[row_idx], dim_k * sizeof(int16_t));
        dot_16(dim_k, row1xk, kx1, &mx1[i]);
    }
    free(row1xk);
}


void *p_mmv(void* arguments) {
    struct p_mmv_args *args = arguments;
    int idx = args->row_idx * args->dim_k;
    mmv(args->dim_m, args->dim_k, &args->mxk[idx], args->kx1, &args->mx1[args->row_idx]);
    free(args);
}

void *p_mmv_32(void* arguments) {
    struct p_mmv_args *args = arguments;
    int idx = args->row_idx * args->dim_k;
    mmv_32(args->dim_m, args->dim_k, &args->mxk_32[idx], args->kx1_32, &args->mx1_32[args->row_idx]);
    free(args);
}

void *p_mmv_16(void* arguments) {
    struct p_mmv_args *args = arguments;
    int idx = args->row_idx * args->dim_k;
    mmv_16(args->dim_m, args->dim_k, &args->mxk_16[idx], args->kx1_16, &args->mx1_16[args->row_idx]);
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

void mmv_pthread_32(int num_thread, int dim_m, int dim_k, int32_t* mxk, int32_t* kx1, int32_t* mx1) {

    pthread_t tid[num_thread];

    int row_sz = dim_m / num_thread;
    struct p_mmv_args args, *_args;
    args.dim_m = row_sz;
    args.dim_k = dim_k;
    args.mxk_32 = mxk;
    args.kx1_32 = kx1;
    args.mx1_32 = mx1;
    args.row_idx = -1;

    for (int t = 0; t < num_thread; t++) {
        _args = (struct p_mmv_args *) malloc(sizeof(struct p_mmv_args));
        memcpy(_args, &args, sizeof(struct p_mmv_args));
        _args->row_idx = t * row_sz;
        pthread_create(tid + t, NULL, p_mmv_32, (void *) _args);
    }

    if (dim_m % num_thread != 0) {
        _args = (struct p_mmv_args *) malloc(sizeof(struct p_mmv_args));
        memcpy(_args, &args, sizeof(struct p_mmv_args));
        _args->row_idx = num_thread * row_sz;
        _args->dim_m = dim_m - num_thread * row_sz;
        pthread_create(tid + num_thread, NULL, p_mmv_32, (void *) _args);
    }

    for (int t = 0; t < num_thread; t++) {
        pthread_join(tid[t], NULL);
    }
}

void mmv_pthread_16(int num_thread, int dim_m, int dim_k, int16_t* mxk, int16_t* kx1, int16_t* mx1) {

    pthread_t tid[num_thread];

    int row_sz = dim_m / num_thread;
    struct p_mmv_args args, *_args;
    args.dim_m = row_sz;
    args.dim_k = dim_k;
    args.mxk_16 = mxk;
    args.kx1_16 = kx1;
    args.mx1_16 = mx1;
    args.row_idx = -1;

    for (int t = 0; t < num_thread; t++) {
        _args = (struct p_mmv_args *) malloc(sizeof(struct p_mmv_args));
        memcpy(_args, &args, sizeof(struct p_mmv_args));
        _args->row_idx = t * row_sz;
        pthread_create(tid + t, NULL, p_mmv_16, (void *) _args);
    }

    if (dim_m % num_thread != 0) {
        _args = (struct p_mmv_args *) malloc(sizeof(struct p_mmv_args));
        memcpy(_args, &args, sizeof(struct p_mmv_args));
        _args->row_idx = num_thread * row_sz;
        _args->dim_m = dim_m - num_thread * row_sz;
        pthread_create(tid + num_thread, NULL, p_mmv_16, (void *) _args);
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

            hwi_idx = (w * i) * _h + i * _w;
            hwoi_idx = (w * o * i) * _h + (o * i) * _w;

            v_i1 = v_hwi + hwi_idx;
            v_oi = v_hwoi + hwoi_idx;

            float *tmp_o = (float *) calloc(o, sizeof(float));
            mmv_pthread(num_thread, o, i, v_oi, v_i1, tmp_o);
            for (int _o = 0; _o < o; _o++) {
                v_o[_o] += tmp_o[_o];
            }
            free(tmp_o);

        }
    }
}

void einsum_hwi_hwoi_to_o_32(int* shape, int32_t* v_hwi, int32_t* v_hwoi, int32_t* v_o, int num_thread) {
    // shape: h, w, o, i
    int h = shape[0];
    int w = shape[1];
    int o = shape[2];
    int i = shape[3];
    int hwi_idx, hwoi_idx;

    int32_t* v_oi;
    int32_t* v_i1;

    for (int _h = 0; _h < h; _h++) {
        for (int _w = 0; _w < w; _w++) {

            hwi_idx = (w * i) * _h + i * _w;
            hwoi_idx = (w * o * i) * _h + (o * i) * _w;

            v_i1 = v_hwi + hwi_idx;
            v_oi = v_hwoi + hwoi_idx;

            int32_t *tmp_o = (int32_t *) calloc(o, sizeof(int32_t));
            mmv_pthread_32(num_thread, o, i, v_oi, v_i1, tmp_o);
            for (int _o = 0; _o < o; _o++) {
                v_o[_o] += tmp_o[_o];
            }
            free(tmp_o);

        }
    }
}

void einsum_hwi_hwoi_to_o_16(int* shape, int16_t* v_hwi, int16_t* v_hwoi, int16_t* v_o, int num_thread) {
    // shape: h, w, o, i
    int h = shape[0];
    int w = shape[1];
    int o = shape[2];
    int i = shape[3];
    int hwi_idx, hwoi_idx;

    int16_t* v_oi;
    int16_t* v_i1;

    for (int _h = 0; _h < h; _h++) {
        for (int _w = 0; _w < w; _w++) {

            hwi_idx = (w * i) * _h + i * _w;
            hwoi_idx = (w * o * i) * _h + (o * i) * _w;

            v_i1 = v_hwi + hwi_idx;
            v_oi = v_hwoi + hwoi_idx;

            int16_t *tmp_o = (int16_t *) calloc(o, sizeof(int16_t));
            mmv_pthread_16(num_thread, o, i, v_oi, v_i1, tmp_o);
            for (int _o = 0; _o < o; _o++) {
                v_o[_o] += tmp_o[_o];
            }
            free(tmp_o);

        }
    }
}


struct Tensor conv2d(struct Tensor input, struct Tensor kernel, int num_thread, int quant_byte) {
    // input.shape: (N, H, W, C=IC)
    // kernel.shape: (KH, KW, OC, IC=C)
    // output.shape: (N, H, W, OC)
    int output_dim = input.shape[0] * input.shape[1] * input.shape[2] * kernel.shape[2];
    struct Tensor out;

    int output_sz;
    if (quant_byte == 32) {
        output_sz = output_dim * sizeof(int32_t);
        out.v32 = (int32_t *) malloc(output_sz);
    } else if (quant_byte == 16) {
        output_sz = output_dim * sizeof(int16_t);
        out.v16 = (int16_t *) malloc(output_sz);
    } else {
        output_sz = output_dim * sizeof(float);
        out.vector = (float *) malloc(output_sz);
    }
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

    int start_odx;
    int oc = kernel.shape[2];
    for (int _n = 0; _n < input.shape[0]; _n++) {  // N
        for (int _h = 0; _h < input.shape[1]; _h++) { // H
            for (int _w = 0; _w < input.shape[2]; _w++) { // W

                start_odx = o0 * _n + o1 * _h + o2 * _w;

                if (quant_byte == 32) {
                    int32_t *v_o = (int32_t *) calloc(oc, sizeof(int32_t));
                    int32_t *patch_hwi = get_input_patch_32(input, pad, _n, _h, _w, kernel.shape[0], kernel.shape[1],
                                                            i0, i1, i2);
                    einsum_hwi_hwoi_to_o_32(kernel.shape, patch_hwi, kernel.v32, v_o, num_thread);
                    memcpy(out.v32 + start_odx, v_o, oc * sizeof(int32_t));
                    free(patch_hwi);
                    free(v_o);
                } else if (quant_byte == 16) {
                    int16_t *v_o = (int16_t *) calloc(oc, sizeof(int16_t));
                    int16_t *patch_hwi = get_input_patch_16(input, pad, _n, _h, _w, kernel.shape[0], kernel.shape[1],
                                                            i0, i1, i2);
                    einsum_hwi_hwoi_to_o_16(kernel.shape, patch_hwi, kernel.v16, v_o, num_thread);
                    memcpy(out.v16 + start_odx, v_o, oc * sizeof(int16_t));
                    free(patch_hwi);
                    free(v_o);
                } else {
                    float *v_o = (float *) calloc(oc, sizeof(float));
                    float *patch_hwi = get_input_patch(input, pad, _n, _h, _w, kernel.shape[0], kernel.shape[1], i0, i1, i2);
                    einsum_hwi_hwoi_to_o(kernel.shape, patch_hwi, kernel.vector, v_o, num_thread);
                    memcpy(out.vector + start_odx, v_o, oc * sizeof(float));
                    free(patch_hwi);
                    free(v_o);
                }
            }
        }
    }
    return out;
}


int main (int argc, char* argv[]) {

    clock_t start, end;
    float elapsed_time;

    clock_t start_q, end_q;
    float elapsed_time_q = 0;

    int quant_byte;
    float quant_const = 10;
    if (!strcmp(argv[3], "FP32")) {
        quant_byte = 0;
    } else if (!strcmp(argv[3], "INT32")) {
        quant_byte = 32;
    } else if (!strcmp(argv[3], "INT16")) {
        quant_byte = 16;
    }

    if (quant_byte == 32) {
        quant_const = 1000;
    } else if (quant_byte == 16) {
        quant_const = 50;
    }

    struct Tensor tensor_in = get_tensor(argv[1]);
    struct Tensor tensor_ke = get_tensor(argv[2]);

    struct Tensor tensor_ot, tensor_quant_ot;
    struct Tensor tensor_in_q, tensor_ke_q;

    int tensor_sz;
    float nrmse;

    int num_thread = 1;

    tensor_ot = conv2d(tensor_in, tensor_ke, num_thread, -1);
    tensor_sz = tensor_ot.shape[0] * tensor_ot.shape[1] * tensor_ot.shape[2] * tensor_ot.shape[3];

    start_q = clock();
    tensor_in_q = quantize(tensor_in, quant_byte, quant_const);
    tensor_ke_q = quantize(tensor_ke, quant_byte, quant_const);
    end_q = clock();
    elapsed_time_q += (float) (end_q - start_q) / CLOCKS_PER_SEC;

    start = clock();
    tensor_quant_ot = conv2d(tensor_in_q, tensor_ke_q, num_thread, quant_byte);
    end = clock();
    elapsed_time = (float) (end - start) / CLOCKS_PER_SEC;

    start_q = clock();
    tensor_quant_ot = recover_tensor(tensor_quant_ot, quant_byte, quant_const);
    end_q = clock();
    elapsed_time_q += (float) (end_q - start_q) / CLOCKS_PER_SEC;
    // for (int i = 0; i < 5; i ++)
    //    printf("%f %f \n", tensor_ot.vector[i], tensor_quant_ot.vector[i]);

    write_tensor(tensor_quant_ot, "output_tensor.bin");

    nrmse = get_nrmse(tensor_quant_ot.vector, tensor_ot.vector, tensor_sz);

    printf("For byte %d, const %f \n", quant_byte, quant_const);
    printf("*  Elapsed time: %f \n", elapsed_time);
    printf("* Overhead time: %f \n", elapsed_time_q);
    printf("*         NRMSE: %f \n", nrmse);

    return 0;
}