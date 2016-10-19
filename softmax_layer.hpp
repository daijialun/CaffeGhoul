#ifndef SOFTMAX_LAYER_HPP_
#define SOFTMAX_LAYER_HPP_

#include <vector>
#include "blob.hpp"
#include "layer.hpp"
namespace caffe  {
class SoftmaxLayer : public Layer  {
public:
        explicit SoftmaxLayer(const LayerParameter& param)
            : Layer(param) {}
        virtual void SetUp(const std::vector<Blob*>& bottom,
            std::vector<Blob*>* top);

        virtual void Forward(const std::vector<Blob*>& bottom, std::vector<Blob*>* top);
        virtual float Backward(const std::vector<Blob*>& top,
            const bool propagate_down, std::vector<Blob*>* bottom);

 protected:
        // sum_multiplier is just used to carry out sum using blas
        Blob sum_multiplier_;
        // scale is an intermediate blob to hold temporary results.
        Blob scale_;
};

}
#endif // SOFTMAX_LAYER_HPP_
