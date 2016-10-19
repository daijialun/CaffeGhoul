#include <algorithm>
#include <cmath>
#include <cfloat>

#include "layer.hpp"
#include "accuracy_layer.hpp"
#include "math_functions.hpp"

namespace caffe  {

void AccuracyLayer::SetUp(const std::vector<Blob*>& bottom, std::vector<Blob*>* top) {
        CHECK_EQ(bottom.size(), 2) << "Accuracy Layer takes two blobs as input.";
        CHECK_EQ(top->size(), 1) << "Accuracy Layer takes 1 output.";
        CHECK_EQ(bottom[0]->num(), bottom[1]->num())
            << "The data and label should have the same number.";
        CHECK_EQ(bottom[1]->channels(), 1);
        CHECK_EQ(bottom[1]->height(), 1);
        CHECK_EQ(bottom[1]->width(), 1);
        (*top)[0]->Reshape(1, 1, 1, 1);
}

void AccuracyLayer::Forward(const std::vector<Blob*>& bottom, std::vector<Blob*>* top) {
        float accuracy = 0;
        const float* bottom_data = bottom[0]->data();
        const float* bottom_label = bottom[1]->data();
        int num = bottom[0]->num();
        int dim = bottom[0]->count() / bottom[0]->num();
        for (int i = 0; i < num; ++i) {
                float maxval = -FLT_MAX;
                int max_id = 0;
                for (int j = 0; j < dim; ++j) {
                        if (bottom_data[i * dim + j] > maxval) {
                                maxval = bottom_data[i * dim + j];
                                max_id = j;
                        }
                }
                if (max_id == (int)bottom_label[i]) {
                        ++accuracy;
                }
        }
        accuracy /= num;
        // LOG(INFO) << "Accuracy: " << accuracy;
        (*top)[0]->mutable_data()[0] = accuracy;
}


}
