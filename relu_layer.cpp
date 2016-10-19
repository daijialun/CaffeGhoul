#include <vector>
#include <algorithm>
#include <glog/logging.h>
#include <cmath>

#include "blob.hpp"
#include "relu_layer.hpp"

namespace caffe {

void ReLULayer::SetUp(const std::vector<Blob*>& bottom, std::vector<Blob*>* top) {
        CHECK_EQ(bottom.size(), 1) << "Neuron Layer takes a single blob as input.";
        CHECK_EQ(top->size(), 1) << "Neuron Layer takes a single blob as output.";
        (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
}

void ReLULayer::Forward(const std::vector<Blob*>& bottom, std::vector<Blob*>* top) {
        const float* bottom_data = bottom[0]->data();
        float* top_data = (*top)[0]->mutable_data();
        const int count = bottom[0]->count();
        for (int i = 0; i < count; ++i) {
                top_data[i] = std::max(bottom_data[i], (float)0);
        }
}

float ReLULayer::Backward(const std::vector<Blob*>& top, const bool propagate_down,
    std::vector<Blob*>* bottom) {
        if (propagate_down) {
                const float* bottom_data = (*bottom)[0]->data();
                const float* top_diff = top[0]->diff();
                float* bottom_diff = (*bottom)[0]->mutable_diff();
                const int count = (*bottom)[0]->count();
                for (int i = 0; i < count; ++i) {
                        bottom_diff[i] = top_diff[i] * (bottom_data[i] > 0);
                }
        }
        return 0;
}
}
