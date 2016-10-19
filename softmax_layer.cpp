#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>
#include <glog/logging.h>

#include "softmax_loss_layer.hpp"
#include "math_functions.hpp"

namespace caffe  {

void SoftmaxLayer::SetUp(const std::vector<Blob*>& bottom,
      std::vector<Blob*>* top) {
        CHECK_EQ(bottom.size(), 1) << "Softmax Layer takes a single blob as input.";
        CHECK_EQ(top->size(), 1) << "Softmax Layer takes a single blob as output.";
        (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
        bottom[0]->height(), bottom[0]->width());
        sum_multiplier_.Reshape(1, bottom[0]->channels(),
        bottom[0]->height(), bottom[0]->width());
        float* multiplier_data = sum_multiplier_.mutable_data();
        for (int i = 0; i < sum_multiplier_.count(); ++i) {
                multiplier_data[i] = 1.;
        }
        scale_.Reshape(bottom[0]->num(), 1, 1, 1);
}

void SoftmaxLayer::Forward(const std::vector<Blob*>& bottom, std::vector<Blob*>* top) {
        const float* bottom_data = bottom[0]->data();
        float* top_data = (*top)[0]->mutable_data();
        float* scale_data = scale_.mutable_data();
        int num = bottom[0]->num();
        int dim = bottom[0]->count() / bottom[0]->num();
        memcpy(top_data, bottom_data, sizeof(float) * bottom[0]->count());
        // we need to subtract the max to avoid numerical issues, compute the exp,
        // and then normalize.
        for (int i = 0; i < num; ++i) {
                scale_data[i] = bottom_data[i*dim];
                for (int j = 0; j < dim; ++j) {
                        scale_data[i] = std::max(scale_data[i], bottom_data[i * dim + j]);
                }
        }
        // subtraction
        caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
            scale_data, sum_multiplier_.data(), 1., top_data);
        // Perform exponentiation
        caffe_exp(num * dim, top_data, top_data);
        // sum after exp
        caffe_cpu_gemv(CblasNoTrans, num, dim, 1., top_data,
            sum_multiplier_.data(), 0., scale_data);
        // Do division
        for (int i = 0; i < num; ++i) {
            caffe_scal(dim, float(1.) / scale_data[i], top_data + i * dim);
        }
}

float SoftmaxLayer::Backward(const std::vector<Blob*>& top,
    const bool propagate_down, std::vector<Blob*>* bottom) {
        const float* top_diff = top[0]->diff();
        const float* top_data = top[0]->data();
        float* bottom_diff = (*bottom)[0]->mutable_diff();
        float* scale_data = scale_.mutable_data();
        int num = top[0]->num();
        int dim = top[0]->count() / top[0]->num();
        memcpy(bottom_diff, top_diff, sizeof(float) * top[0]->count());
        // Compute inner1d(top_diff, top_data) and subtract them from the bottom diff
        for (int i = 0; i < num; ++i) {
            scale_data[i] = caffe_cpu_dot(dim, top_diff + i * dim, top_data + i * dim);
        }
        // subtraction
        caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
            scale_data, sum_multiplier_.data(), 1., bottom_diff);
        // elementwise multiplication
        caffe_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
        return float(0);
}



}

