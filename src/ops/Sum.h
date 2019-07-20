//
// Created by polarbabe on 22.05.19.
//

#ifndef LIBDL_SUM_H
#define LIBDL_SUM_H


#include "../Tensor.h"
#include "../Utils.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

template <typename D, int R>
class Sum : public CNode<D, 0> {
public:
    Sum(const std::optional<std::shared_ptr<CNode<D, R>>> &a,
        const std::shared_ptr<Tensor<D, 0>> &r,
        const std::array<long, R> &shape)
            : CNode<D, 0>(Utils::removeOption<std::shared_ptr<CNodeBase>>({a}), r), a(a), shape(shape) {}

    static std::shared_ptr<Tensor<D, 0>> sum(const std::shared_ptr<Tensor<D, R>> &a) {
        auto result = std::make_shared<Tensor<D, 0>>(a->eTensor->sum(), std::array<long, 0> {});
        if (a->needsGradient())
            result->setGradFn(std::make_shared<Sum<D, R>>(a->gradFn, result, a->eTensor->dimensions()));
        return result;
    }

    void computeGradients() override {
        if (a.has_value()) {
            /*
             broadcasting an tensor of dimension 1 is buggy

             Eigen::Tensor<float, 1> t(1);
             t.setConstant(3);
             Eigen::Tensor<float, 1> t2 = t.reshape(Eigen::array<long, 1> {1}).broadcast(Eigen::array<long, 1> {16});
             std::cout << t2 << std::endl;

            */
            std::array<int, R> r;
            r.fill(1);
            a.value()->addGrad(CNode<D, 0>::grad->reshape(r).broadcast(shape));
        }
        CNode<D, 0>::finishComputeGradient();
    }

private:
    std::optional<std::shared_ptr<CNode<D, R>>> a;
    std::array<long, R> shape;
};

#endif //LIBDL_SUM_H
