#include <string>
#include <vector>
#include  <glog/logging.h>
#include <boost/shared_ptr.hpp>
#include <map>
#include <set>

#include "caffe.pb.h"
#include "net.hpp"
#include "layer_factory.hpp"


namespace caffe {

Net::Net(const NetParameter& param, const std::vector<Blob*> &bottom)  {
        // Basically, build all the layers and set up its connections.
        name_ = param.name();
        std::map<std::string, int> blob_name_to_idx;
        std::set<std::string> available_blobs;
        int num_layers = param.layers_size();
        CHECK_EQ(bottom.size(), param.input_size())
            << "Incorrect bottom blob size.";
        // set the input blobs
        for (int i = 0; i < param.input_size(); ++i) {
                const std::string& blob_name = param.input(i);
                CHECK_GT(bottom[i]->count(), 0);
                boost::shared_ptr<Blob> blob_pointer( new Blob(bottom[i]->num(),
                    bottom[i]->channels(), bottom[i]->height(), bottom[i]->width()));
                blobs_.push_back(blob_pointer);
                blob_names_.push_back(blob_name);
                net_input_blob_indices_.push_back(i);
                blob_name_to_idx[blob_name] = i;
                available_blobs.insert(blob_name);
        }

        // For each layer, set up their input and output
        bottom_vecs_.resize(param.layers_size());
        top_vecs_.resize(param.layers_size());
        bottom_id_vecs_.resize(param.layers_size());
        top_id_vecs_.resize(param.layers_size());
        for (int i = 0; i < param.layers_size(); ++i) {
                const LayerConnection& layer_connection = param.layers(i);
                const LayerParameter& layer_param = layer_connection.layer();
                layers_.push_back(boost::shared_ptr<Layer>(GetLayer(layer_param)));
                layer_names_.push_back(layer_param.name());
                LOG(INFO) << "Creating Layer " << layer_param.name();

                // Figure out this layer's input and output
                for (int j = 0; j < layer_connection.bottom_size(); ++j) {
                        const std::string& blob_name = layer_connection.bottom(j);
                        if (available_blobs.find(blob_name) == available_blobs.end()) {
                                LOG(FATAL) << "Unknown blob input " << blob_name << " to layer" << j;
                        }
                        LOG(INFO) << layer_param.name() << " <- " << blob_name;
                        bottom_vecs_[i].push_back(
                        blobs_[blob_name_to_idx[blob_name]].get());
                        bottom_id_vecs_[i].push_back(blob_name_to_idx[blob_name]);
                        available_blobs.erase(blob_name);
                }
                for (int j = 0; j < layer_connection.top_size(); ++j) {
                        const std::string& blob_name = layer_connection.top(j);
                        if (blob_name_to_idx.find(blob_name) != blob_name_to_idx.end()) {
                                LOG(FATAL) << "Duplicate blobs produced by multiple sources.";
                        }
                        LOG(INFO) << layer_param.name() << " -> " << blob_name;
                        boost::shared_ptr<Blob> blob_pointer(new Blob());
                        blobs_.push_back(blob_pointer);
                        blob_names_.push_back(blob_name);
                        blob_name_to_idx[blob_name] = blob_names_.size() - 1;
                        available_blobs.insert(blob_name);
                        top_vecs_[i].push_back(blobs_[blob_names_.size() - 1].get());
                        top_id_vecs_[i].push_back(blob_names_.size() - 1);
                }
        }

        // In the end, all remaining blobs are considered output blobs.
        for (std::set<std::string>::iterator it = available_blobs.begin(); it != available_blobs.end(); ++it) {
                LOG(ERROR) << "This network produces output " << *it;
                net_output_blob_indices_.push_back(blob_name_to_idx[*it]);
                net_output_blobs_.push_back(blobs_[blob_name_to_idx[*it]].get());
        }

        LOG(ERROR) << "Setting up the layers.";
        for (int i = 0; i < layers_.size(); ++i) {
                LOG(INFO) << "Setting up " << layer_names_[i];
                layers_[i]->SetUp(bottom_vecs_[i], &top_vecs_[i]);
                std::vector<boost::shared_ptr<Blob> >& layer_blobs = layers_[i]->blobs();
                for (int j = 0; j < layer_blobs.size(); ++j) {
                        params_.push_back(layer_blobs[j]);
                }
                for (int topid = 0; topid < top_vecs_[i].size(); ++topid) {
                        LOG(INFO) << "Top shape: " << top_vecs_[i][topid]->channels() << " "
                          << top_vecs_[i][topid]->height() << " "
                          << top_vecs_[i][topid]->width();
                }
        }
        LOG(ERROR) << "Network initialization done.";
}

}

