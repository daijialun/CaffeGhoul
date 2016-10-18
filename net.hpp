#ifndef CAFFE_NET_HPP_
#define CAFFE_NET_HPP_

#include <vector>
#include <string>
#include <boost/shared_ptr.hpp>
#include "layer.hpp"
#include "caffe.pb.h"

namespace caffe {
class Net {
public:
        Net( const NetParameter& param, const std::vector<Blob*> &bottom);
        ~Net() {}
        const std::vector<Blob*>& Forward(const std::vector<Blob*> &bottom);
        float Backward();
        float ForwardBackward(const std::vector<Blob*>& bottom)  {
                Forward(bottom);
                return Backward();
        }
protected:
        // Individual layers in the net
        std::vector<boost::shared_ptr<Layer> > layers_;
        std::vector<std::string> layer_names_;
        // Intermediate results between layers
        std::vector<boost::shared_ptr<Blob> > blobs_;
        std::vector<std::string> blob_names_;
        // Input vector for each layer
        std::vector<std::vector<Blob*> > bottom_vecs_;
        std::vector<int> bottom_id_vecs_;
        // Output vector for each layer
        std::vector<std::vector<Blob*> > top_vecs_;
        std::vector<int> top_id_vecs_;
        // Output of network
        std::vector<Blob*> net_output_blobs_;
        // Name of network
        std::string name_;
        // parameters in the network
        std::vector<boost::shared_ptr<Blob> > param_;

};
}
#endif // CAFFE_NET_HPP_
