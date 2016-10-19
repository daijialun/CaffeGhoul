#ifndef SOFTMAX_LOSS_LAYER_HPP_
#define SOFTMAX_LOSS_LAYER_HPP_

#include <vector>
#include <boost/shared_ptr.hpp>
#include "blob.hpp"
#include "layer.hpp"
#include "softmax_layer.hpp"

namespace caffe  {
class SoftmaxWithLossLayer : public Layer  {
public:
        explicit SoftmaxWithLossLayer(const LayerParameter& param)
            : Layer(param) {}
        virtual void SetUp(const std::vector<Blob*>& bottom,
            std::vector<Blob*>* top);

        virtual void Forward(const std::vector<Blob*>& bottom,
            std::vector<Blob*>* top);
        virtual float Backward(const std::vector<Blob*>& top,
            const bool propagate_down, std::vector<Blob*>* bottom);

 protected:
        boost::shared_ptr<SoftmaxLayer> softmax_layer_;
        // prob stores the output probability of the layer.
        Blob prob_;
        // Vector holders to call the underlying softmax layer forward and backward.
        std::vector<Blob*> softmax_bottom_vec_;
        std::vector<Blob*> softmax_top_vec_;
};
}
#endif // SOFTMAX_LOSS_LAYER_HPP_
