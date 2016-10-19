#ifndef RELU_LAYER_HPP_
#define RELU_LAYER_HPP_

#include <vector>
#include "layer.hpp"

namespace caffe {

class ReLULayer : public Layer {
public:
        explicit ReLULayer(const LayerParameter& param) : Layer(param) {}
        void SetUp(const std::vector<Blob*>& bottom, std::vector<Blob*>* top);
        virtual void Forward(const std::vector<Blob*>& bottom, std::vector<Blob*>* top);
        virtual float Backward(const std::vector<Blob*>& top,
            const bool propagate_down, std::vector<Blob*>* bottom);
};

}
#endif // RELU_LAYER_HPP_
