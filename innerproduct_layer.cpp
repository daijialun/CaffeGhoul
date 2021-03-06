#include <vector>
#include <glog/logging.h>
#include "blob.hpp"
#include "layer.hpp"
#include "innerproduct_layer.hpp"
#include "math_functions.hpp"

namespace caffe  {

void InnerProductLayer::SetUp(const std::vector<Blob*>& bottom, std::vector<Blob*>* top) {
        CHECK_EQ(bottom.size(), 1) << "IP Layer takes a single blob as input.";
        CHECK_EQ(top->size(), 1) << "IP Layer takes a single blob as output.";
        const int num_output = this->layer_param_.num_output();
        biasterm_ = this->layer_param_.biasterm();
        // Figure out the dimensions
        M_ = bottom[0]->num();
        K_ = bottom[0]->count() / bottom[0]->num();
        N_ = num_output;
        (*top)[0]->Reshape(bottom[0]->num(), num_output, 1, 1);
        // Check if we need to set up the weights
        if (this->blobs_.size() > 0) {
                LOG(INFO) << "Skipping parameter initialization";
        } else {
                /*if (biasterm_) {
                        this->blobs_.resize(2);
                } else {
                        this->blobs_.resize(1);
                }
                // Intialize the weight
                this->blobs_[0].reset(new Blob<float>(1, 1, N_, K_));
                // fill the weights
                shared_ptr<Filler<float> > weight_filler(
                GetFiller<float>(this->layer_param_.weight_filler()));
                weight_filler->Fill(this->blobs_[0].get());
                // If necessary, intiialize and fill the bias term
                if (biasterm_) {
                        this->blobs_[1].reset(new Blob<float>(1, 1, 1, N_));
                        shared_ptr<Filler<float> > bias_filler(
                        GetFiller<float>(this->layer_param_.bias_filler()));
                        bias_filler->Fill(this->blobs_[1].get());
                }*/
        } // parameter initialization
        // Setting up the bias multiplier
        /*if (biasterm_) {
                bias_multiplier_.reset(new SyncedMemory(M_ * sizeof(float)));
                float* bias_multiplier_data =
                reinterpret_cast<float*>(bias_multiplier_->mutable_cpu_data());
                for (int i = 0; i < M_; ++i) {
                        bias_multiplier_data[i] = 1.;
                }
        }*/
};

void InnerProductLayer::Forward(const std::vector<Blob*>& bottom, std::vector<Blob*>* top) {
        const float* bottom_data = bottom[0]->data();
        float* top_data = (*top)[0]->mutable_data();
        const float* weight = this->blobs_[0]->data();
        caffe_cpu_gemm(CblasNoTrans, CblasTrans, M_, N_, K_, 1,
            bottom_data, weight, 0, top_data);
        /*if (biasterm_) {
                caffe_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (float)1.,
                    reinterpret_cast<const float*>(bias_multiplier_->cpu_data()),
                    this->blobs_[1]->cpu_data(), (float)1., top_data);
        }*/
}

float InnerProductLayer::Backward(const std::vector<Blob*>& top,
    const bool propagate_down, std::vector<Blob*>* bottom) {
        const float* top_diff = top[0]->diff();
        const float* bottom_data = (*bottom)[0]->data();
        // Gradient with respect to weight
        caffe_cpu_gemm(CblasTrans, CblasNoTrans, N_, K_, M_, (float)1.,
            top_diff, bottom_data, (float)0., this->blobs_[0]->mutable_diff());
        /*if (biasterm_) {
                // Gradient with respect to bias
                caffe_cpu_gemv<float>(CblasTrans, M_, N_, (float)1., top_diff,
                    reinterpret_cast<const float*>(bias_multiplier_->cpu_data()), (float)0.,
                    this->blobs_[1]->mutable_cpu_diff());
        }*/
        if (propagate_down) {
                // Gradient with respect to bottom data
                caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, M_, K_, N_, (float)1.,
                    top_diff, this->blobs_[0]->data(), (float)0.,
                        (*bottom)[0]->mutable_diff());
        }
        return float(0);
}

}
