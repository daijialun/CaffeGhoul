#include <string>
#include <vector>
#include  <glog/logging.h>
#include <

#include "caffe.pb.h"
#include "net.hpp"


namespace caffe {

Net::Net(const NetParameter& param, const std::vector<Blob*> &bottom)  {
        name_=param.name();
        int num_layer = param.layers_size();
        CHECK_EQ(bottom.size(), param.input_size())
                    << "Incorrect bottom blob size.";
        for(int i=0; i<param.input_size(); i++)  {
                const std::string& blob_name = param.input(i);
                CHECK_GT(bottom[i]->count(), 0);
                shared_ptr<Blob<Dtype> > blob_pointer( new Blob(bottom[i]->num(), bottom[i]->channels(),
                                bottom[i]->height(), bottom[i]->width()));
                blobs_.push_back(blob_pointer);
                blob_names_.push_back(blob_name);
        }
        bottom_vecs_.resize(param.layers_size());
        top_vecs_.resize(param.layers_size());
        for (int i = 0; i < param.layers_size(); ++i) {

        }
}
}

