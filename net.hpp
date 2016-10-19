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

        // returns the network name.
        inline const std::string& name() { return name_; }
        // returns the layer names
        inline const std::vector<std::string>& layer_names() { return layer_names_; }
        // returns the blob names
        inline const std::vector<std::string>& blob_names() { return blob_names_; }
        // returns the blobs
        inline const std::vector<boost::shared_ptr<Blob> >& blobs() { return blobs_; }
        // returns the layers
        inline const std::vector<boost::shared_ptr<Layer> >& layers() { return layers_; }
        // returns the bottom and top vecs for each layer - usually you won't need
        // this unless you do per-layer checks such as gradients.
        inline std::vector<std::vector<Blob*> >& bottom_vecs() { return bottom_vecs_; }
        inline std::vector<std::vector<Blob*> >& top_vecs() { return top_vecs_; }
        // returns the parameters
        std::vector<boost::shared_ptr<Blob > >& params() { return params_; };

        void Update();
        void ToProto(NetParameter* param, bool write_diff = false);

protected:
        // Individual layers in the net
        std::vector<boost::shared_ptr<Layer> > layers_;
        std::vector<std::string> layer_names_;
        // Intermediate results between layers
        std::vector<boost::shared_ptr<Blob> > blobs_;
        std::vector<std::string> blob_names_;
        // Input vector for each layer
        std::vector<std::vector<Blob*> > bottom_vecs_;
        std::vector<std::vector<int> > bottom_id_vecs_;
        // Output vector for each layer
        std::vector<std::vector<Blob*> > top_vecs_;
        std::vector<std::vector<int> >  top_id_vecs_;
        // Output of network
        std::vector<int> net_input_blob_indices_;
        std::vector<int> net_output_blob_indices_;
        std::vector<Blob*> net_output_blobs_;
        // Name of network
        std::string name_;
        // parameters in the network
        std::vector<boost::shared_ptr<Blob> > params_;

};
}
#endif // CAFFE_NET_HPP_
