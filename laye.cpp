#include "layer.hpp"

namespace caffe  {

void Layer::ToProto(LayerParameter* param, bool write_diff) {
        param->Clear();
        param->CopyFrom(layer_param_);
        param->clear_blobs();
        for (int i = 0; i < blobs_.size(); ++i) {
                blobs_[i]->ToProto(param->add_blobs(), write_diff);
        }
}

}
