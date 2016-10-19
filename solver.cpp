#include <iostream>
#include <fstream>
#include <string>
#include <glog/logging.h>

#include "solver.hpp"
#include "blob.hpp"
#include "net.hpp"
#include "io.hpp"
#include "math_functions.hpp"

namespace caffe {
void Solver::Solve(Net* net) {
        net_ = net;
        LOG(INFO) << "Solving " << net_->name();
        PreSolve();
        iter_ = 0;
        // For a network that is trained by the solver, no bottom or top vecs
        // should be given, and we will just provide dummy vecs.
        std::vector<Blob*> bottom_vec;
        while (iter_++ < param_.max_iter()) {
                float loss = net_->ForwardBackward(bottom_vec);
                ComputeUpdateValue();
                net_->Update();

                // Check if we need to do snapshot
                /*if (param_.snapshot() > 0 && iter_ % param_.snapshot() == 0) {
                        Snapshot(false);
                }*/
                if (param_.display()) {
                        LOG(ERROR) << "Iteration " << iter_ << ", loss = " << loss;
                }
        }
        LOG(INFO) << "Optimization Done.";
}

void SGDSolver::PreSolve() {
        // First of all, see if we need to initialize the history
        std::vector<boost::shared_ptr<Blob> >& net_params = this->net_->params();
        history_.clear();
        for (int i = 0; i < net_params.size(); ++i) {
                const Blob* net_param = net_params[i].get();
                history_.push_back(boost::shared_ptr<Blob>(new Blob(
                    net_param->num(), net_param->channels(), net_param->height(),
                    net_param->width())));
        }
}

void SGDSolver::ComputeUpdateValue() {
        std::vector<boost::shared_ptr<Blob> >& net_params = this->net_->params();
        // get the learning rate
        float rate = GetLearningRate();
        float momentum = this->param_.momentum();
        float weight_decay = this->param_.weight_decay();
        // LOG(ERROR) << "rate:" << rate << " momentum:" << momentum
        //     << " weight_decay:" << weight_decay;
        for (int param_id = 0; param_id < net_params.size(); ++param_id) {
                // Compute the value to history, and then copy them to the blob's diff.
                caffe_axpby(net_params[param_id]->count(), rate,
                net_params[param_id]->diff(), momentum,
                history_[param_id]->mutable_data());
                if (weight_decay) {
                        // add weight decay
                        caffe_axpy(net_params[param_id]->count(), weight_decay * rate,
                        net_params[param_id]->data(),
                        history_[param_id]->mutable_data());
                }
                // copy
                caffe_copy(net_params[param_id]->count(),
                history_[param_id]->data(),
                net_params[param_id]->mutable_diff());
    }
}

}
