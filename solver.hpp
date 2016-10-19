#ifndef CAFFE_OPTIMIZATION_SOLVER_HPP_
#define CAFFE_OPTIMIZATION_SOLVER_HPP_

#include <vector>
#include<boost/shared_ptr.hpp>
#include "caffe.pb.h"
#include "net.hpp"

namespace caffe  {

class Solver {
public:
        explicit Solver(const SolverParameter& param) : param_(param) {}
        void Solve(Net*  net);

protected:
        virtual void PreSolve() {};
        // Get the update value for the current iteration.
        virtual void ComputeUpdateValue() = 0;
        SolverParameter param_;
        Net* net_;
        int iter_;
};

class SGDSolver : public Solver {
public:
        explicit SGDSolver(const SolverParameter& param) : Solver(param) {}

protected:
        virtual void PreSolve();
        float GetLearningRate();
        virtual void ComputeUpdateValue();
        // history maintains the historical momentum data.
        std::vector<boost::shared_ptr<Blob> > history_;

};

}

#endif // CAFFE_OPTIMIZATION_SOLVER_HPP_
