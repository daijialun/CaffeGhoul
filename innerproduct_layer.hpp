#ifndef INNERPRODUCT_LAYER_HPP_
#define INNERPRODUCT_LAYER_HPP_

#include <boost/shared_ptr.hpp>
#include <vector>
#include "layer.hpp"
#include "caffe.pb.h"

namespace caffe  {

class InnerProductLayer : public Layer {
public:
        explicit InnerProductLayer(const LayerParameter& param) : Layer(param) {}
        virtual void SetUp(const std::vector<Blob*>& bottom, std::vector<Blob*>* top);
        virtual void Forward(const std::vector<Blob*>& bottom,
            std::vector<Blob*>* top);
        virtual float Backward(const std::vector<Blob*>& top,
            const bool propagate_down, std::vector<Blob*>* bottom);

protected:
        int M_;
        int K_;
        int N_;
        bool biasterm_;
        boost::shared_ptr<float> bias_multiplier_;
};

}
#endif // INNERPRODUCT_LAYER_HPP_
