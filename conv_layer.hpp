#ifndef CONV_LAYER_HPP_
#define CONV_LAYER_HPP_

#include <vector>
#include <boost/shared_ptr.hpp>
#include "layer.hpp"
#include "caffe.pb.h"

namespace caffe {
class ConvolutionLayer : public  Layer  {
public:
        explicit ConvolutionLayer(const LayerParameter& param) : Layer(param)  {}
        virtual void SetUp(const std::vector<Blob*>& bottom, std::vector<Blob*>* top);
        virtual void Forward(const std::vector<Blob*>& bottom, std::vector<Blob*>* top);
        virtual float Backward(const std::vector<Blob*> &top,
                               const bool propagate_down, std::vector<Blob*> *bottom);

protected:
        Blob col_bob_;
        boost::shared_ptr<float> bias_multiplier_;
        int KSIZE_;
        int STRIDE_;
        int NUM_;
        int CHANNELS_;
        int HEIGHT_;
        int WIDTH_;
        int NUM_OUTPUT_;
        int GROUP_;
        Blob col_buffer_;
        bool biasterm_;
        int M_;
        int K_;
        int N_;
};
}
#endif // CONV_LAYER_HPP_
