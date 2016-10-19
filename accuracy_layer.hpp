#ifndef ACCURACY_LAYER_HPP_
#define ACCURACY_LAYER_HPP_

#include <vector>
#include "blob.hpp"
#include "layer.hpp"

namespace caffe  {
class AccuracyLayer : public Layer {
public:
        explicit AccuracyLayer(const LayerParameter& param) : Layer(param) {}
        virtual void SetUp(const std::vector<Blob*>& bottom, std::vector<Blob*>* top);
        virtual void Forward(const std::vector<Blob*>& bottom, std::vector<Blob*>* top);
        // The accuracy layer should not be used to compute backward operations.
        virtual float Backward(const std::vector<Blob*>& top,
            const bool propagate_down, std::vector<Blob*>* bottom) {
                    return float(0.);
        }

};

}
#endif // ACCURACY_LAYER_HPP_
