#ifndef POOLING_LAYER_HPP_
#define POOLING_LAYER_HPP_

#include <vector>
#include "blob.hpp"
#include "layer.hpp"
#include "caffe.pb.h"

namespace caffe {

class PoolingLayer : public Layer {
public:
        explicit PoolingLayer(const LayerParameter& param) : Layer(param) {}
        virtual void SetUp(const std::vector<Blob*> &bottom, std::vector<Blob*> *top);
        virtual void Forward(const std::vector<Blob*> &bottom, std::vector<Blob*> *top);
        virtual float Backward(const std::vector<Blob*> &top, const bool propagate_down, std::vector<Blob*> *bottom);

protected:
        int KSIZE_;
        int STRIDE_;
        int CHANNELS_;
        int HEIGHT_;
        int WIDTH_;
        int POOLED_HEIGHT_;
        int POOLED_WIDTH_;
};
}
#endif // POOLING_LAYER_HPP_
