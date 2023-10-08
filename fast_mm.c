#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>
#include <time.h>

#define N 1000 // Adjust the matrix size as needed

void matrix_multiply(float* A, float* B, float* C, int n) {
    // #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float cij = 0.0;
            for (int k = 0; k < n; k++) {
                cij += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = cij;
        }
    }
}

void matrix_multiply_faster(float* A, float* B, float* C, int m, int n, int k) {
    #pragma omp parallel for
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            __m256 sum = _mm256_setzero_ps();
            for (int i = 0; i < k; i += 8) {
                __m256 a = _mm256_load_ps(&A[row * k + i]);
                __m256 b = _mm256_load_ps(&B[col * k + i]);
                sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
            }
            float result[8];
            _mm256_storeu_ps(result, sum);
            for (int j = 0; j < 8; j++) {
                C[row * n + col] += result[j];
            }
        }
    }
}

int main() {
    int length = 4096;
    int hid_dim = 64;
    
    float* A = (float*)aligned_alloc(32, length * hid_dim * sizeof(float)); // 32-byte alignment
    float* B = (float*)aligned_alloc(32, hid_dim * hid_dim * sizeof(float));
    float* C = (float*)aligned_alloc(32, length * hid_dim * sizeof(float));

    // float* A = (float*)malloc(N * N * sizeof(float));
    // float* B = (float*)malloc(N * N * sizeof(float));
    // float* C = (float*)malloc(N * N * sizeof(float));

    // Initialize A and B with your data

    // Use compiler optimizations
    // #pragma omp parallel for
    for (int i = 0; i < length * hid_dim; i++) {
        A[i] = 1.0; // Example initialization
    }
    for (int i = 0; i < hid_dim * hid_dim; i++) {
        B[i] = 2.0; // Example initialization
    }

    clock_t start, end;
    double cpu_time_used;
    time_t wall_time_start, wall_time_end;

    start = clock();
    time(&wall_time_start);

    // Matrix multiplication with optimizations
    for (int i = 0; i < 100; ++i) {
    // matrix_multiply(A, B, C, N);
        matrix_multiply_faster(A, B, C, length, hid_dim, hid_dim);
    }
    
    end = clock();
    time(&wall_time_end);
    
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    double wall_time_used = difftime(wall_time_end, wall_time_start);
    
    printf("CPU time: %f ms\n", cpu_time_used  * 1000 / 100);
    printf("Wall time: %f ms\n", wall_time_used * 1000 / 100);
    // Use the resulting matrix C

    free(A);
    free(B);
    free(C);

    return 0;
}
