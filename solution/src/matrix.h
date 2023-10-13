#include <immintrin.h>
#include <assert.h>
#include <math.h>

#ifndef MATRIX_H    
#define MATRIX_H

typedef struct Matrix {
    float* data;
    int m;
    int n;
    int stride;
} Matrix;

Matrix createMatrix(int m, int n);
int matSize(Matrix mat);
int matSizeBytes(Matrix mat);
void freeMatrix(Matrix mat);

Matrix submatrix(Matrix mat, int r1, int r2, int c1, int c2);

void matFillZeros(Matrix mat);
void matInplaceScalarProd(Matrix mat, float scalar);
void matInplaceSigmoid(Matrix a);
void matInplaceTanh(Matrix a);
void matCopy(Matrix a, Matrix b);
void matSum(Matrix a, Matrix b, Matrix c);
void matHProduct(Matrix a, Matrix b, Matrix c);
void matVecProduct(Matrix a, Matrix b, Matrix c);
void matSlerp(Matrix a, Matrix b, Matrix t, Matrix c);
int vecArgmax(Matrix a);

#endif