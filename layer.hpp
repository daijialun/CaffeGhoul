#ifndef CAFFE_LAYER_HPP_
#define CAFFE_LAYER_HPP_

#include <vector>
#include <boost/shared_ptr.hpp>
#include "blob.hpp"
#include "caffe.pb.h"

namespace caffe {
class Layer  {
public:
         explicit Layer(const LayerParameter& param) : layer_param_(param) { }
         virtual ~Layer() {}
         void Reshape(const int num, const int height, const int width, const int channels);
         virtual void SetUp(const std::vector<Blob*> &bottom, std::vector<Blob*> *top) = 0;
         virtual void Forward(const std::vector<Blob*> &bottom, std::vector<Blob*> *top)=0;
         virtual float Backward(const std::vector<Blob*> &top, std::vector<Blob*> *bottom)=0;

protected:
        LayerParameter layer_param_;
        std::vector<boost::shared_ptr<Blob> > blobs_;
};
}
#endif // CAFFE_LAYER_HPP_
