#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>
#include <glog/logging.h>

#include "softmax_loss_layer.hpp"
#include "math_functions.hpp"

namespace caffe  {

void SoftmaxWithLossLayer::SetUp(const std::vector<Blob*>& bottom,
      std::vector<Blob*>* top) {
        CHECK_EQ(bottom.size(), 2) << "SoftmaxLoss Layer takes a single blob as input.";
        CHECK_EQ(top->size(), 0) << "SoftmaxLoss Layer takes no blob as output.";
        softmax_bottom_vec_.clear();
        softmax_bottom_vec_.push_back(bottom[0]);
        softmax_top_vec_.push_back(&prob_);
        softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);
}

void SoftmaxWithLossLayer::Forward(const std::vector<Blob*>& bottom,
    std::vector<Blob*>* top) {
        // The forward pass computes the softmax prob values.
        softmax_bottom_vec_[0] = bottom[0];
        softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
}

float SoftmaxWithLossLayer::Backward(const std::vector<Blob*>& top,
    const bool propagate_down,
        std::vector<Blob*>* bottom) {
        // First, compute the diff
        float* bottom_diff = (*bottom)[0]->mutable_diff();
        const float* prob_data = prob_.data();
        memcpy(bottom_diff, prob_data, sizeof(float) * prob_.count());
        const float* label = (*bottom)[1]->data();
        int num = prob_.num();
        int dim = prob_.count() / num;
        float loss = 0;
        for (int i = 0; i < num; ++i) {
                bottom_diff[i * dim + static_cast<int>(label[i])] -= 1;
                loss += -log(std::max(prob_data[i * dim + static_cast<int>(label[i])], FLT_MIN));
        }
        // Scale down gradient
        caffe_scal(prob_.count(), float(1) / num, bottom_diff);
        return loss / num;
}

}
