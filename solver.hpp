#ifndef CAFFE_OPTIMIZATION_SOLVER_HPP_
#define CAFFE_OPTIMIZATION_SOLVER_HPP_
class Solver {
public:
        explicit Solver(const SolverParameter& param) : param_(param) {}

protected:
        SolverParameter param_;
        Net* net_;
        int iter_;

};

#endif // CAFFE_OPTIMIZATION_SOLVER_HPP_
