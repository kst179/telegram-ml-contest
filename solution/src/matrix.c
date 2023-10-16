#include <math.h>
#include <stdlib.h>

#include "matrix.h"

Matrix createMatrix(int m, int n) {
    assert(n % 8 == 0);

    Matrix mat; 
    mat.data = aligned_alloc(32, m * n * sizeof(float));
    mat.m = m;
    mat.n = n;
    mat.stride = n;

    return mat;
}

void freeMatrix(Matrix mat) {
    free(mat.data);
}

int matSize(Matrix mat) { return mat.n * mat.m; }
int matSizeBytes(Matrix mat) { return mat.n * mat.m * sizeof(float); }

Matrix submatrix(Matrix mat, int r1, int r2, int c1, int c2) {
    Matrix submat;// = malloc(sizeof(Matrix));

    assert(c1 % 8 == 0);
    assert(c2 % 8 == 0);

    submat.data = mat.data + r1 * mat.stride + c1;
    submat.m = r2 - r1;
    submat.n = c2 - c1;

    return submat;
}

void matFillZeros(Matrix mat) {
    __m256 zeros = _mm256_setzero_ps();

    for (int i = 0; i < mat.m * mat.n; i+=8) {
        int row = i / mat.n;
        int col = i % mat.n;

        _mm256_store_ps(&mat.data[row * mat.stride + col], zeros);
    }
}

void matInplaceScalarProd(Matrix mat, float scalar) {
    __m256 vec_scalar = _mm256_set1_ps(scalar); 

    for (int i = 0; i < mat.m * mat.n; i += 8) {
        int row = i / mat.n;
        int col = i % mat.n;
        __m256 vec_mat = _mm256_load_ps(&mat.data[row * mat.stride + col]);
        vec_mat = _mm256_mul_ps(vec_mat, vec_scalar);
        _mm256_store_ps(&mat.data[row * mat.stride + col], vec_mat);
    }
}

void matCopy(Matrix a, Matrix b) {
    for (int i = 0; i < a.m * a.n; i+=8) {
        int row = i / a.n;
        int col = i % a.n;
        __m256 vec_a = _mm256_load_ps(&a.data[row * a.stride + col]);
        _mm256_store_ps(&b.data[row * b.stride + col], vec_a);
    }
}

void matSum(Matrix a, Matrix b, Matrix c) {
    for (int i = 0; i < a.m * a.n; i+=8) {
        int row = i / a.n;
        int col = i % a.n;
        __m256 vec_a = _mm256_load_ps(&a.data[row * a.stride + col]);
        __m256 vec_b = _mm256_load_ps(&b.data[row * b.stride + col]);
        __m256 vec_c = _mm256_add_ps(vec_a, vec_b);
        _mm256_store_ps(&c.data[row * c.stride + col], vec_c);
    }
}

void matHProduct(Matrix a, Matrix b, Matrix c) { 
    for (int i = 0; i < a.m * a.n; i+=8) {
        int row = i / a.n;
        int col = i % a.n;
        __m256 vec_a = _mm256_load_ps(&a.data[row * a.stride + col]);
        __m256 vec_b = _mm256_load_ps(&b.data[row * b.stride + col]);
        __m256 vec_c = _mm256_mul_ps(vec_a, vec_b);
        _mm256_store_ps(&c.data[row * c.stride + col], vec_c);
    }
}

/*
 * c = a * (1 - t) + b * t
 */
void matSlerp(Matrix a, Matrix b, Matrix t, Matrix c) {
    __m256 vec_ones = _mm256_set1_ps(1.0f);

    for (int i = 0; i < a.m * a.n; i+=8) {
        int row = i / a.n;
        int col = i % a.n;
        __m256 vec_a = _mm256_load_ps(&a.data[row * a.stride + col]);
        __m256 vec_b = _mm256_load_ps(&b.data[row * b.stride + col]);
        __m256 vec_t = _mm256_load_ps(&t.data[row * t.stride + col]);

        vec_a = _mm256_mul_ps(_mm256_sub_ps(vec_ones, vec_t), vec_a);
        vec_a = _mm256_add_ps(vec_a, _mm256_mul_ps(vec_t, vec_b));

        _mm256_store_ps(&c.data[row * c.stride + col], vec_a);
    }
}

void matInplaceSigmoid(Matrix a) {
    for (int i = 0; i < a.m * a.n; i++) {
        int row = i / a.n;
        int col = i % a.n;

        float x = a.data[row * a.stride + col];
        a.data[row * a.stride + col] = 1 / (1 + exp(-x));
    }
}

void matInplaceTanh(Matrix a) {
    for (int i = 0; i < a.m * a.n; i++) {
        int row = i / a.n;
        int col = i % a.n;

        float x = a.data[row * a.stride + col];
        float e = exp(-2*x);
        a.data[row * a.stride + col] = (1 - e) / (1 + e);
    }
}

void matVecProduct(Matrix a, Matrix b, Matrix c) {
    // assert(b.m == 1);
    float result[8] __attribute__((aligned(32)));

    for (int row = 0; row < a.m; row++) {
        __m256 sum = _mm256_setzero_ps();

        for (int col = 0; col < a.n; col += 8) {
            __m256 vec_a = _mm256_load_ps(&a.data[row * a.stride + col]);
            __m256 vec_b = _mm256_load_ps(&b.data[col]);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(vec_a, vec_b));
        }

        _mm256_store_ps(result, sum);

        for (int i = 0; i < 8; i++) {
            c.data[row] += result[i];
        }
    }
}

int vecArgmax(Matrix a) {
    // assert(a.m = 1);

    float max_val = a.data[0];
    int argmax = 0;

    for (int i = 0; i < a.n; i++) {
        float val = a.data[i];

        if (max_val < val) {
            max_val = val;
            argmax = i;
        }
    }

    return argmax;
}

