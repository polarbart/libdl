#ifndef LIBDL_SUB_H
#define LIBDL_SUB_H

#include "Tensor.h"
#include "CNode.h"
#include "Utils.h"

template <typename D, int RA, int RB>
class Sub : public CNode<D, std::max(RA, RB)>  {

public:
    Sub(std::optional<std::shared_ptr<CNode<D, RA>>> a, std::optional<std::shared_ptr<CNode<D, RB>>> b, std::weak_ptr<Tensor<D, std::max(RA, RB)>> t)
            : CNode<D, std::max(RA, RB)>(Utils::removeOption<std::shared_ptr<CNodeBase>>({a, b}), t), a(a), b(b) {};

    static std::shared_ptr<Tensor<D, std::max(RA, RB)>> sub(std::shared_ptr<Tensor<D, RA>> a, std::shared_ptr<Tensor<D, RB>> b) {
        if constexpr (RB > RA)
            return Sub<D, RB, RA>::sub(b, a);
        else {
            auto data = std::shared_ptr<D[]>(new D[a->eTensor.size()]);
            Eigen::TensorMap<Eigen::Tensor<D, RA>> t(data.get(), a->eTensor.dimensions());
            if constexpr (RB < RA) {
                std::array<int, RA> reshape{};
                std::array<int, RA> broadcast{};
                for (int i = 0; i < RA - RB; ++i) {
                    reshape[i] = 1;
                    broadcast[i] = a->eTensor.dimension(i);
                }
                for (int i = RA - RB; i < RA; ++i) {
                    reshape[i] = b->eTensor.dimension(i - RB);
                    broadcast[i] = 1;
                }
                t = a->eTensor - b->eTensor.reshape(reshape).broadcast(broadcast);
            } else
                t = a->eTensor - b->eTensor;
            std::array<long, RA> shape{};
            std::copy(t.dimensions().begin(), t.dimensions().end(), shape.begin());
            auto result = std::make_shared<Tensor<D, RA>>(data, shape);
            if (a->needsGradient() || b->needsGradient())
                result->setGradFn(std::make_shared<Sub<D, RA, RB>>(a->gradFn, b->gradFn, result));
            return result;
        }
    }

    void computeGradients() override {
        if (a.has_value())
            a.value()->addGrad(CNode<D, RA>::grad);
        if (b.has_value()) {
            if constexpr (RB < RA) {
                std::array<int, RA - RB> sum{};
                for (int i = 0; i < RA - RB; ++i)
                    sum[i] = i;
                Eigen::Tensor<D, RB> x = -CNode<D, RA>::grad.sum(sum);
                b.value()->addGrad(x);
            } else {
                Eigen::Tensor<D, RB> x = -CNode<D, RA>::grad;
                b.value()->addGrad(x);
            }
        }
        CNode<D, RA>::finishComputeGradient();
    }

private:
    std::optional<std::shared_ptr<CNode<D, RA>>> a;
    std::optional<std::shared_ptr<CNode<D, RB>>> b;
};

#endif //LIBDL_SUB_H
