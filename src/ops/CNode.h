
#ifndef LIBDL_CNODE_H
#define LIBDL_CNODE_H
#define EIGEN_USE_THREADS

#include <memory>
#include <vector>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CNodeBase.h"

template <typename D, int R>
class Tensor;

template <typename D, int R>
class CNode : public CNodeBase {
public:

    void emptyGrad() {
        grad = std::make_shared<Eigen::Tensor<D, R>>(shape);
        resetGrad = false;
    }

    template <typename Derived>
    void addGrad(const Derived &g) {
        static Eigen::ThreadPool pool(8);
        static Eigen::ThreadPoolDevice myDevice(&pool, 8);
        if (resetGrad) {
            if (grad.use_count() == 0)
                grad = std::make_shared<Eigen::Tensor<D, R>>(shape);
            grad->device(myDevice) = g;
            resetGrad = false;
        } else
            grad->device(myDevice) += g;
    }

    void zeroGrad() {
        resetGrad = true;
    }

    std::shared_ptr<Eigen::Tensor<D, R>> grad;
    const std::array<long, R> shape;

protected:
    CNode(const std::vector<std::shared_ptr<CNodeBase>>& p, const std::shared_ptr<Tensor<D, R>> &holder) : CNodeBase(p), shape(holder->data->dimensions()), holder(holder) {}

    void finishComputeGradient() {
        if (holder.expired())
            return;
        auto p = holder.lock();
        if (p->requiresGrad)
            p->addGrad(grad);
        p->gradFn = std::nullopt;
    }

    std::weak_ptr<Tensor<D, R>> holder;
    bool resetGrad = true;
};


#endif //LIBDL_CNODE_H
