
#ifndef LIBDL_MATMUL_H
#define LIBDL_MATMUL_H

#include <memory>
#include "CNode.h"
#include "../Tensor.h"
#include "../Utils.h"

template <typename D, std::int64_t RA, std::int64_t RB>
class MatMul : public CNode<D, RA + RB - 2> {

public:
    MatMul(
            const std::shared_ptr<Tensor<D, RA>> &a,
            const std::shared_ptr<Tensor<D, RB>> &b,
            const std::shared_ptr<Tensor<D, RA + RB - 2>> &result)
            : CNode<D, RA + RB - 2>(Utils::removeOption<std::shared_ptr<CNodeBase>>({a->gradFn, b->gradFn}), result),
            a(a->data),
            b(b->data),
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

        if (a->data->dimension(RA - 1) != b->data->dimension(0))
            throw std::invalid_argument("the last dimension of a and the first dimension of b must match");

        std::array<std::int64_t, RA + RB - 2> shape {};
        std::copy_n(std::begin(a->data->dimensions()), RA - 1, std::begin(shape));
        std::copy_n(std::begin(b->data->dimensions()) + 1, RB - 1, std::begin(shape) + RA - 1);

        auto t = a->data->contract(*b->data, Eigen::array<Eigen::IndexPair <std::int64_t>, 1>{Eigen::IndexPair <std::int64_t>(RA-1, 0)});
        auto result = std::make_shared<Tensor<D, RA + RB - 2>>(t, shape);

        if ((a->needsGradient() || b->needsGradient()) && !CNodeBase::noGrad)
            result->setGradFn(std::make_shared<MatMul<D, RA, RB>>(a, b, result));
        return result;
    }

    void computeGradients() override {
        if (ca.has_value()) {
            Eigen::array<Eigen::IndexPair <std::int64_t>, RB - 1> d {};
            for (std::int64_t i = 0; i < RB - 1; ++i)
                d[i] = Eigen::IndexPair <std::int64_t>(RA + RB - 2 - i - 1, RB - i - 1);
            ca.value()->addGrad(CNode<D, RA + RB - 2>::grad->contract(*b, d));
        }
        if (cb.has_value()) {
            Eigen::array<Eigen::IndexPair <std::int64_t>, RA - 1> d {};
            for (std::int64_t i = 0; i < RA - 1; ++i)
                d[i] = Eigen::IndexPair <std::int64_t>(i, i);
            cb.value()->addGrad(a->contract(*CNode<D, RA + RB - 2>::grad, d));
        }
        CNode<D, RA + RB - 2>::finishComputeGradient();
    }

private:
    std::shared_ptr<Eigen::Tensor<D, RA>> a;
    std::shared_ptr<Eigen::Tensor<D, RB>> b;
    std::optional<std::shared_ptr<CNode<D, RA>>> ca;
    std::optional<std::shared_ptr<CNode<D, RB>>> cb;
};

#endif //LIBDL_MATMUL_H
