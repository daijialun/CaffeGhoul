#include <iostream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <map>
#include <string>
#include <vector>

#include "caffe.pb.h"
#include "blob.hpp"
#include "io.hpp"
#include "net.hpp"
#include "caffe.pb.h"
#include "solver.hpp"
/*
DEFINE_string(solver, "",
    "The solver definition protocol buffer text file.");

DEFINE_string(model, "",
    "The model definition protocol buffer text file.");

DEFINE_int32(iterations, 50,
    "The number of iterations to run.");

typedef int (*BrewFunction)();
typedef std::map<std::string, BrewFunction> BrewMap;
BrewMap g_brew_map;

// Register a mode: train or test
// #表示输出字符串，##表示连接两个参数
#define RegisterBrewFunction(func) \
class  __Registerer_##func { \
public: \
        __Registerer_##func()  { \
        g_brew_map[#func]=&func; \
        } \
}; \
__Registerer_##func g_registerer_##func;

static BrewFunction GetBrewFunction(const std::string& name)  {
            if( g_brew_map.count(name) )
                    return g_brew_map[name];
            else {
                    LOG(ERROR) << "Available caffe actions:";
                    for (BrewMap::iterator it = g_brew_map.begin();
                         it != g_brew_map.end(); ++it) {
                      LOG(ERROR) << "\t" << it->first;
                    }
                    LOG(FATAL) << "Unknown action: " << name;
                    return NULL;  // not reachable, just to suppress old compiler warnings.
            }
}

int train()  {
            CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
            caffe::NetParameter net_param;
            caffe::ReadProtoFromTextFile("data/lenet.prototxt", &net_param);
            return 0;
}
RegisterBrewFunction(train);

int test()  {
            CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
            return 0;
}
RegisterBrewFunction(test);
*/

int main(int argc, char** argv)
{
        caffe::NetParameter net_param;
        caffe::ReadProtoFromTextFile("data/lenet.prototxt", &net_param);
        std::vector<caffe::Blob*> bottom_vec;
        caffe::Net caffe_net(net_param, bottom_vec);

        LOG(ERROR) << "Performing Forward";
        caffe_net.Forward(bottom_vec);
        LOG(ERROR) << "Performing Backward";
        LOG(ERROR) << "Initial loss: " << caffe_net.Backward();

        caffe::SolverParameter solver_param;
        solver_param.set_base_lr(0.01);
        solver_param.set_display(1);
        solver_param.set_max_iter(6000);
        solver_param.set_lr_policy("inv");
        solver_param.set_gamma(0.0001);
        solver_param.set_power(0.75);
        solver_param.set_momentum(0.9);
        solver_param.set_weight_decay(0.0005);


        LOG(ERROR) << "Starting Optimization";
        caffe::SGDSolver solver(solver_param);
        solver.Solve(&caffe_net);
        LOG(ERROR) << "Optimization Done.";


        // Run the network after training.
        LOG(ERROR) << "Performing Forward";
        caffe_net.Forward(bottom_vec);
        LOG(ERROR) << "Performing Backward";
        float loss = caffe_net.Backward();
        LOG(ERROR) << "Final loss: " << loss;

        caffe::NetParameter trained_net_param;
        caffe_net.ToProto(&trained_net_param);

        return 0;
}
