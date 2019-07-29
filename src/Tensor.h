
#ifndef LIBDL_TENSOR_H
#define LIBDL_TENSOR_H
#define EIGEN_USE_THREADS

#include <iostream>
#include <memory>
#include <unsupported/Eigen/CXX11/Tensor>
#include <queue>
#include "ops/CNode.h"
#include "ops/Leaf.h"
#include "GlobalThreadPool.h"

template <typename D, int R>
class Tensor {
public:

    std::shared_ptr<Eigen::Tensor<D, R>> data;
    std::shared_ptr<Eigen::Tensor<D, R>> grad;
    std::optional<std::shared_ptr<CNode<D, R>>> gradFn;
    bool requiresGrad;

    template<typename OtherDerived>
    Tensor(const OtherDerived &t, const std::array<long, R> &shape)
            : data(std::make_shared<Eigen::Tensor<D, R>>(shape)),
              gradFn(std::nullopt),
              requiresGrad(false) {
                data->device(GlobalThreadPool::myDevice) = t;
              }

    Tensor(Eigen::Tensor<D, R> t, bool requiresGrad)
            : data(std::make_shared<Eigen::Tensor<D, R>>(std::move(t))),
              gradFn(std::nullopt),
              requiresGrad(requiresGrad) {}

    explicit Tensor(const std::array<long, R> &shape, bool requiresGrad = false)
            : data(std::make_shared<Eigen::Tensor<D, R>>(shape)),
              gradFn(std::nullopt), // TODO leaf node
              requiresGrad(requiresGrad) {}

    void setGradFn(const std::shared_ptr<CNode<D, R>>& g) {
        if (!CNodeBase::noGrad)
            gradFn = std::optional<std::shared_ptr<CNode<D, R>>>(g);
    }

    bool needsGradient() {
        return requiresGrad || gradFn.has_value();
    }

    void zeroGrad() {
        grad = std::shared_ptr<Eigen::Tensor<D, R>>(nullptr);
    }

    // subtract this tensors gradient from this tensor (used for gradient decent)
    void subGrad(D lr) {
        if (grad.use_count() > 0)
            data->device(GlobalThreadPool::myDevice) -= grad->constant(lr) * *grad;
    }

    // add the given tensor onto this tensors gradient (used for backpropagation)
    void addGrad(const std::shared_ptr<Eigen::Tensor<D, R>> &g) {
        if (grad.use_count() == 0) {
            grad = g;
        } else if (grad != g)
            grad->device(GlobalThreadPool::myDevice) += *g;
    }

    // backpropagation algorithm
    void backward(D v = 1) {
        // neither this tensor nor any of its predecessors needs a gradient
        if (!gradFn.has_value())
            return;

        gradFn.value()->addGrad(data->constant(v));

        auto head = gradFn.value();

        // first go through the graph
        std::queue<std::shared_ptr<CNodeBase>> q;
        q.push(head);
        while (!q.empty()) {
            auto e = q.front();
            q.pop();
            for (const auto& n : e->children) {
                n->parentsThatNeedToComputeGradients++;
                if (!n->visited) {
                    n->visited = true;
                    q.push(n);
                }
            }
        }

        q.push(head);
        while (!q.empty()) {
            auto e = q.front();
            q.pop();
            if (e->parentsThatNeedToComputeGradients > 0) {
                q.push(e);
                continue;
            }
            e->computeGradients();
            for (const auto& n : e->children) {
                n->parentsThatNeedToComputeGradients--;
                if (n->visited) {
                    n->visited = false;
                    q.push(n);
                }
            }
        }
    }
};





#endif //LIBDL_TENSOR_H
