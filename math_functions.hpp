#ifndef CAFFE_MATH_FUNCTIONS_HPP_
#define CAFFE_MATH_FUNCTIONS_HPP_

#include <boost/math/special_functions/next.hpp>
#include <glog/logging.h>
extern "C" {
#include <cblas.h>
}

#include "mkl_alternate.hpp"

namespace caffe {
void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C);

void caffe_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const float alpha, const float* A, const float* x, const float beta,
    float* y);

void caffe_axpy(const int N, const float alpha, const float* X, float* Y);

void caffe_scal(const int N, const float alpha, float *X);

void caffe_exp(const int n, const float* a, float* y);

void caffe_mul(const int N, const float* a, const float* b, float* y);

float caffe_cpu_dot(const int n, const float* x, const float* y);
/*void caffe_cpu_axpby(const int N, const float alpha, const float* X,
    const float beta, float* Y);

void caffe_copy(const int N, const float *X, float *Y);

void caffe_set(const int N, const float alpha, float *X);


inline void caffe_memset(const size_t N, const int alpha, void* X) {
  memset(X, alpha, N);  // NOLINT(caffe/alt_fn)
}

void caffe_add_scalar(const int N, const float alpha, float *X);

void caffe_scal(const int N, const float alpha, float *X);

void caffe_sqr(const int N, const float* a, float* y);

void caffe_add(const int N, const float* a, const float* b, float* y);

void caffe_sub(const int N, const float* a, const float* b, float* y);

template <typename float>
void caffe_mul(const int N, const float* a, const float* b, float* y);

template <typename float>
void caffe_div(const int N, const float* a, const float* b, float* y);

template <typename float>
void caffe_powx(const int n, const float* a, const float b, float* y);

unsigned int caffe_rng_rand();

float caffe_nextafter(const float b);

void caffe_rng_uniform(const int n, const float a, const float b, float* r);

void caffe_rng_gaussian(const int n, const float mu, const float sigma, float* r);

void caffe_rng_bernoulli(const int n, const float p, int* r);

void caffe_rng_bernoulli(const int n, const float p, unsigned int* r);

void caffe_exp(const int n, const float* a, float* y);

void caffe_log(const int n, const float* a, float* y);

void caffe_abs(const int n, const float* a, float* y);

float caffe_cpu_dot(const int n, const float* x, const float* y);

float caffe_cpu_strided_dot(const int n, const float* x, const int incx, const float* y, const int incy);

int caffe_cpu_hamming_distance(const int n, const float* x, const float* y);

float caffe_cpu_asum(const int n, const Dtype* x);*/
}

#endif // CAFFE_MATH_FUNCTIONS_HPP_
