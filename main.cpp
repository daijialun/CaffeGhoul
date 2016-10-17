#include <iostream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <map>
#include <string>
#include<vector>

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
            NetParameter net_param;
            ReadProtoFromTextFile("data/lenet.prototxt", &net_param);
            ReadSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param);
            return 0;
}
RegisterBrewFunction(train);

int test()  {
            CHECK_GT(FLAGS_solver.size(), 0) << "Need a solver definition to train.";
            return 0;
}
RegisterBrewFunction(train);

int main(int argc, char** argv)
{
        FLAGS_alsologtostderr = 1;
        GlobalInit(&argc, &argv);
        GetBrewFunction(std::string(argv[1]))();
        return 0;
}
