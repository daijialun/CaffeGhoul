#include <vector>
#include "conv_layer.hpp"
#include "im2col.hpp"
#include "math_functions.hpp"

namespace caffe {

void ConvolutionLayer::SetUp(const std::vector<Blob*> &bottom, std::vector<Blob*> *top)  {
        CHECK_EQ(bottom.size(), 1) << "Conv Layer takes a single blob as input.";
        CHECK_EQ(top->size(), 1) << "Conv Layer takes a single blob as output.";
        KSIZE_ = this->layer_param_.kernelsize();
        STRIDE_ = this->layer_param_.stride();
        GROUP_ = this->layer_param_.group();
        NUM_ = bottom[0]->num();
        CHANNELS_ = bottom[0]->channels();
        HEIGHT_ = bottom[0]->height();
        WIDTH_ = bottom[0]->width();
        NUM_OUTPUT_ = this->layer_param_.num_output();
        CHECK_GT(NUM_OUTPUT_, 0);
        CHECK_EQ(CHANNELS_ % GROUP_, 0);
        // im2col result buffer
        int height_out = (HEIGHT_ - KSIZE_) / STRIDE_ + 1;
        int width_out = (WIDTH_ - KSIZE_) / STRIDE_ + 1;
        col_buffer_.Reshape(1, CHANNELS_ * KSIZE_ * KSIZE_, height_out, width_out);
        // Set the parameters
        CHECK_EQ(NUM_OUTPUT_ % GROUP_, 0)
                << "Number of output should be multiples of group.";
        biasterm_ = this->layer_param_.biasterm();
        // Figure out the dimensions for individual gemms.
        M_ = NUM_OUTPUT_ / GROUP_;
        K_ = CHANNELS_ * KSIZE_ * KSIZE_ / GROUP_;
        N_ = height_out * width_out;
        (*top)[0]->Reshape(bottom[0]->num(), NUM_OUTPUT_, height_out, width_out);
         if (this->blobs_.size() > 0) {
                LOG(INFO) << "Skipping parameter initialization";
          } else {
                if (biasterm_) {
                        this->blobs_.resize(2);
                } else {
                        this->blobs_.resize(1);
                }
                // Intialize the weight
                this->blobs_[0].reset(
                    new Blob(NUM_OUTPUT_, CHANNELS_ / GROUP_, KSIZE_, KSIZE_));
               /* // fill the weights
                shared_ptr<Filler<Dtype> > weight_filler(
                    GetFiller<Dtype>(this->layer_param_.weight_filler()));
                weight_filler->Fill(this->blobs_[0].get());  */
                // If necessary, intiialize and fill the bias term
                if (biasterm_) {
                  this->blobs_[1].reset(new Blob(1, 1, 1, NUM_OUTPUT_));
                /*  shared_ptr<Filler<Dtype> > bias_filler(
                      GetFiller<Dtype>(this->layer_param_.bias_filler()));
                  bias_filler->Fill(this->blobs_[1].get());*/
                }
        }
        // Set up the bias filler
        if (biasterm_) {
                bias_multiplier_.reset(new float[N_] );
/*        boost::shared_ptr<float> bias_multiplier_data = bias_multiplier_->get();
        for (int i = 0; i < N_; ++i) {
                bias_multiplier_data[i] = 1.;
        }*/
        }
}

void ConvolutionLayer::Forward(const std::vector<Blob*>& bottom, std::vector<Blob*>* top)  {file:///home/white_ghoul/%E6%A1%8C%E9%9D%A2/CaffeGhoul/caffe.pb.h

        const float*bottom_data = bottom[0]->data();
        float* top_data = (*top)[0]->mutable_data();
        float* col_data = col_buffer_.mutable_data();
        const float* weight = this->blobs_[0]->data();
        int weight_offset = M_ * K_;
        int col_offset = K_ * N_;
        int top_offset = M_ * N_;
        for (int n = 0; n < NUM_; ++n) {
                // First, im2col
                im2col_cpu(bottom_data + bottom[0]->offset(n), CHANNELS_, HEIGHT_,
                    WIDTH_, KSIZE_, STRIDE_, col_data);
                // Second, innerproduct with groups
                for (int g = 0; g < GROUP_; ++g) {
                        caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, M_, N_, K_,
                                1., weight + weight_offset * g, col_data + col_offset * g,
                                0., top_data + (*top)[0]->offset(n) + top_offset * g);
                }
                // third, add bias
                /*if (biasterm_) {
                        caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, NUM_OUTPUT_,
                            N_, 1, 1., this->blobs_[1]->data(),
                           (const float*)bias_multiplier_->get(),
                            1., top_data + (*top)[0]->offset(n));
                }*/
        }
}

float ConvolutionLayer::Backward(const std::vector<Blob*> &top, const bool propagate_down, std::vector<Blob*> *bottom) {
        const float* top_diff = top[0]->diff();
        const float* weight = this->blobs_[0]->data();
        float* weight_diff = this->blobs_[0]->mutable_diff();
        const float* bottom_data = (*bottom)[0]->data();
        float* bottom_diff = (*bottom)[0]->mutable_diff();
        float* col_data = col_buffer_.mutable_data();
        float* col_diff = col_buffer_.mutable_diff();
        // bias gradient if necessary
        float* bias_diff = NULL;

        /*if (biasterm_) {
                bias_diff = this->blobs_[1]->mutable_cpu_diff();
                memset(bias_diff, 0., sizeof(Dtype) * this->blobs_[1]->count());
                for (int n = 0; n < NUM_; ++n) {
                    caffe_cpu_gemv<Dtype>(CblasNoTrans, NUM_OUTPUT_, N_,
                    1., top_diff + top[0]->offset(n),
                    reinterpret_cast<const Dtype*>(bias_multiplier_->cpu_data()), 1.,
                    bias_diff);
                }
        }*/

        int weight_offset = M_ * K_;
        int col_offset = K_ * N_;
        int top_offset = M_ * N_;
        memset(weight_diff, 0., sizeof(float) * this->blobs_[0]->count());
        for (int n = 0; n < NUM_; ++n) {
        // since we saved memory in the forward pass by not storing all col data,
        // we will need to recompute them.
        im2col_cpu(bottom_data + (*bottom)[0]->offset(n), CHANNELS_, HEIGHT_,
            WIDTH_, KSIZE_, STRIDE_, col_data);
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        /*for (int g = 0; g < GROUP_; ++g) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
        (Dtype)1., top_diff + top[0]->offset(n) + top_offset * g,
        col_data + col_offset * g, (Dtype)1.,
        weight_diff + weight_offset * g);
        }*/
        // gradient w.r.t. bottom data, if necessary
        /*if (propagate_down) {
                for (int g = 0; g < GROUP_; ++g) {
                        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
                            (Dtype)1., weight + weight_offset * g,
                            top_diff + top[0]->offset(n) + top_offset * g,
                            (Dtype)0., col_diff + col_offset * g);
        }
        // col2im back to the data
        col2im_cpu(col_diff, CHANNELS_, HEIGHT_,
            WIDTH_, KSIZE_, STRIDE_, bottom_diff + (*bottom)[0]->offset(n));
        }*/
  }
  return 0;
}
}
