
#ifndef LIBDL_MATMUL_H
#define LIBDL_MATMUL_H

#include <memory>
#include "CNode.h"
#include "../Tensor.h"
#include "../Utils.h"

template <typename D, int RA, int RB>
class MatMul : public CNode<D, RA + RB - 2> {

public:
    MatMul(
            const std::shared_ptr<Tensor<D, RA>> &a,
            const std::shared_ptr<Tensor<D, RB>> &b,
            const std::shared_ptr<Tensor<D, RA + RB - 2>> &result)
            : CNode<D, RA + RB - 2>(Utils::removeOption<std::shared_ptr<CNodeBase>>({a->gradFn, b->gradFn}), result),
            a(a->eTensor),
            b(b->eTensor),
            ca(a->gradFn),
            cb(b->gradFn) {}

    /*
     * \brief performs a contraction along the last dimension of a and the first dimension of b
     *
     * \param a a tensor of shape (d_1, ..., d_n, d) and any dimension
     * \param b a tensor of shape (d, e_1, ..., e_m) and any dimension
     *
     * \returns a new tensor of shape (d_1, ..., d_n, e_1, ..., e_n)
     * */
    static std::shared_ptr<Tensor<D, RA + RB - 2>> matmul(
            const std::shared_ptr<Tensor<D, RA>> &a,
            const std::shared_ptr<Tensor<D, RB>> &b) {

        if (a->eTensor->dimension(RA - 1) != b->eTensor->dimension(0))
            throw std::invalid_argument("the last dimension of a and the first dimension of b must match");

        std::array<long, RA + RB - 2> shape {};
        std::copy_n(std::begin(a->eTensor->dimensions()), RA - 1, std::begin(shape));
        std::copy_n(std::begin(b->eTensor->dimensions()) + 1, RB - 1, std::begin(shape) + RA - 1);

        auto t = a->eTensor->contract(*b->eTensor, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(RA-1, 0)});
        auto result = std::make_shared<Tensor<D, RA + RB - 2>>(t, shape);

        if (a->needsGradient() || b->needsGradient())
            result->setGradFn(std::make_shared<MatMul<D, RA, RB>>(a, b, result));
        return result;
    }

    void computeGradients() override {
        if (ca.has_value()) {
            Eigen::array<Eigen::IndexPair<int>, RB - 1> d {};
            for (int i = 0; i < RB - 1; ++i)
                d[i] = Eigen::IndexPair<int>(RA + RB - 2 - i - 1, RB - i - 1);
            ca.value()->addGrad(CNode<D, RA + RB - 2>::grad->contract(*b, d));
        }
        if (cb.has_value()) {
            Eigen::array<Eigen::IndexPair<int>, RA - 1> d {};
            for (int i = 0; i < RA - 1; ++i)
                d[i] = Eigen::IndexPair<int>(i, i);
            cb.value()->addGrad(a->contract(*CNode<D, RA + RB - 2>::grad, d));
        }
        CNode<D, RA + RB - 2>::finishComputeGradient();
    }

private:
    std::shared_ptr<Eigen::TensorMap<Eigen::Tensor<D, RA>>> a;
    std::shared_ptr<Eigen::TensorMap<Eigen::Tensor<D, RB>>> b;
    std::optional<std::shared_ptr<CNode<D, RA>>> ca;
    std::optional<std::shared_ptr<CNode<D, RB>>> cb;
};

#endif //LIBDL_MATMUL_H
