#ifndef LIBDL_SUB_H
#define LIBDL_SUB_H

#include "../Tensor.h"
#include "CNode.h"
#include "../Utils.h"

template <typename D, int RA, int RB>
class Sub : public CNode<D, std::max(RA, RB)>  {

public:
    Sub(
            const std::optional<std::shared_ptr<CNode<D, RA>>> &ca,
            const std::optional<std::shared_ptr<CNode<D, RB>>> &cb,
            const std::shared_ptr<Tensor<D, std::max(RA, RB)>> &t)
            : CNode<D, std::max(RA, RB)>(Utils::removeOption<std::shared_ptr<CNodeBase>>({ca, cb}), t),
            ca(ca),
            cb(cb) {};

    /*
     * \brief Elementwise substraction.
     *        One tensor x can have a less dimensions than the other tensor y, as long as the first dimensions match.
     *        x will be broadcasted to match y.
     *        E.g. shapes (10, 5) and (10, 5, 8) are ok, but shapes (10, 5) and (8, 10, 5) are not
     *
     * \param a tensor of any dimension
     * \param b tensor of any dimension
     *
     * \return a new tensor of the shape with more dimensions
     * */
    static std::shared_ptr<Tensor<D, std::max(RA, RB)>> sub(
            std::shared_ptr<Tensor<D, RA>> a,
            std::shared_ptr<Tensor<D, RB>> b) {

        std::shared_ptr<Tensor<D, std::max(RA, RB)>> result;

        if constexpr (RA == RB)
            result = std::make_shared<Tensor<D, RA>>(*a->eTensor - *b->eTensor, a->eTensor->dimensions());
        else if constexpr (RA < RB)
            result = std::make_shared<Tensor<D, std::max(RA, RB)>>(broadcast(*a->eTensor, *b->eTensor) - *b->eTensor, b->eTensor->dimensions());
        else if constexpr (RA > RB)
            result = std::make_shared<Tensor<D, std::max(RA, RB)>>(*a->eTensor - broadcast(*b->eTensor, *a->eTensor), a->eTensor->dimensions());

        if (a->needsGradient() || b->needsGradient())
            result->setGradFn(std::make_shared<Sub<D, RA, RB>>(a->gradFn, b->gradFn, result));
        return result;
    }

    void computeGradients() override {
        if (ca.has_value()) {
            if constexpr (RA < RB)
                ca.value()->addGrad(sum(*CNode<D, RB>::grad));
            else
                ca.value()->addGrad(*CNode<D, RA>::grad);
        }
        if (cb.has_value()) {
            if constexpr (RB < RA)
                cb.value()->addGrad(-sum(*CNode<D, RA>::grad));
            else
                cb.value()->addGrad(-(*CNode<D, RB>::grad));
        }
        CNode<D, std::max(RA, RB)>::finishComputeGradient();
    }

private:
    std::optional<std::shared_ptr<CNode<D, RA>>> ca;
    std::optional<std::shared_ptr<CNode<D, RB>>> cb;

    static auto broadcast(const Eigen::TensorMap<Eigen::Tensor<D, std::min(RA, RB)>> &b, const Eigen::TensorMap<Eigen::Tensor<D, std::max(RA, RB)>> &a) {
        std::array<int, std::max(RA, RB)> reshape{};
        std::array<int, std::max(RA, RB)> broadcast{};
        for (int i = 0; i < std::min(RA, RB); ++i) {
            reshape[i] = b.dimension(i);
            broadcast[i] = 1;
        }
        for (int i = std::min(RA, RB); i < std::max(RA, RB); ++i) {
            reshape[i] = 1;
            broadcast[i] = a.dimension(i);
        }
        return b.reshape(reshape).broadcast(broadcast);
    }

    static auto sum(const Eigen::TensorMap<Eigen::Tensor<D, std::max(RA, RB)>> &x) {
        std::array<int, std::max(RA, RB) - std::min(RA, RB)> sum {};
        for (int i = 0; i < std::max(RA, RB) - std::min(RA, RB); ++i)
            sum[i] = std::max(RA, RB) + i;
        return x.sum(sum);
    }
};

#endif //LIBDL_SUB_H
