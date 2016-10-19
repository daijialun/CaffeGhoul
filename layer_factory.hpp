#ifndef LAYER_FACTORY_HPP_
#define LAYER_FACTORY_HPP_

#include <string>

#include "layer.hpp"
#include "accuracy_layer.hpp"
#include "conv_layer.hpp"
#include "pooling_layer.hpp"
#include "relu_layer.hpp"
#include "innerproduct_layer.hpp"
#include "softmax_layer.hpp"
#include "softmax_loss_layer.hpp"
#include "caffe.pb.h"

namespace caffe  {

Layer* GetLayer(const LayerParameter& param)  {
        const std::string& type = param.type();
        if (type == "accuracy") {
                Layer *tmp = new AccuracyLayer(param);
                return tmp;
        } else if (type == "conv") {
                return new ConvolutionLayer(param);
        //} else if (type == "data") {
        //        return new DataLayer<Dtype>(param);
        //} else if (type == "dropout") {
        //        return new DropoutLayer<Dtype>(param);
        //} else if (type == "euclidean_loss") {
        //              return new EuclideanLossLayer<Dtype>(param);
//        } else if (type == "im2col") {
 //               return new Im2colLayer(param);
        } else if (type == "innerproduct") {
                return new InnerProductLayer(param);
        //} else if (type == "lrn") {
        //    return new LRNLayer<Dtype>(param);
        //  } else if (type == "padding") {
        //    return new PaddingLayer<Dtype>(param);
        } else if (type == "pool") {
                return new PoolingLayer(param);
        } else if (type == "relu") {
                return new ReLULayer(param);
        } else if (type == "softmax") {
                return new SoftmaxLayer(param);
        } else if (type == "softmax_loss") {
                return new SoftmaxWithLossLayer(param);
        //} else if (type == "multinomial_logistic_loss") {
        //   return new MultinomialLogisticLossLayer<Dtype>(param);
        } else {
                LOG(FATAL) << "Unknown layer name: " << type;
        }
        // just to suppress old compiler warnings.
        return (Layer*)(NULL);
}
}

#endif // LAYER_FACTORY_HPP_
