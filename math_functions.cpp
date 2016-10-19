#include <math.h>
#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>
#include <limits>

#include "math_functions.hpp"

namespace caffe {
void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
        int lda = (TransA == CblasNoTrans) ? K : M;
        int ldb = (TransB == CblasNoTrans) ? N : K;
        cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
}

void caffe_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

void caffe_axpy(const int N, const float alpha, const float* X, float* Y) {
        cblas_saxpy(N, alpha, X, 1, Y, 1);
}

void caffe_scal(const int N, const float alpha, float *X) {
        cblas_sscal(N, alpha, X, 1);
}

void caffe_exp(const int n, const float* a, float* y) {
        vsExp(n, a, y);
}

void caffe_mul(const int n, const float* a, const float* b, float* y) {
        vsMul(n, a, b, y);
}

float caffe_cpu_dot(const int n, const float* x, const float* y) {
        return cblas_sdot(n, x, 1, y, 1);
}

void caffe_copy(const int N, const float *X, float *Y)  {
  cblas_scopy(N, X, 1, Y, 1);
}

void caffe_axpby(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

}





