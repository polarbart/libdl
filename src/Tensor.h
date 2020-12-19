
#ifndef LIBDL_TENSOR_H
#define LIBDL_TENSOR_H
#define EIGEN_USE_THREADS

#include <iostream>
#include <memory>
#include <unsupported/Eigen/CXX11/Tensor>
#include <queue>
#include <optional>
#include "ops/CNode.h"
#include "ops/Leaf.h"
#include "GlobalThreadPool.h"

/*
 * This class represents a tensor, including autograd abilities.
 * I.e. it has a reference to its gradient, its corresponding computational node and a flag if it requires a gradient.
 *
 * the function backward() computes the gradient of all predecessors to this tensor
 *
 * The data Eigen::Tensor as well as the gradient are wrapped into shared pointers since they are also referenced from the CNode class.
 *
 * D and R are the template parameter for Eigen::Tensor
 * */
template <typename D,std::int64_t R>
class Tensor {
public:

    std::shared_ptr<Eigen::Tensor<D, R>> data;
    std::shared_ptr<Eigen::Tensor<D, R>> grad;
    std::optional<std::shared_ptr<CNode<D, R>>> gradFn;
    bool requiresGrad;

    template<typename OtherDerived>
    Tensor(const OtherDerived &t, const std::array<std::int64_t, R> &shape)
            : data(std::make_shared<Eigen::Tensor<D, R>>(shape)),
              gradFn(std::nullopt),
              requiresGrad(false) {
                data->device(GlobalThreadPool::myDevice) = t;
              }

    Tensor(Eigen::Tensor<D, R> t, bool requiresGrad)
            : data(std::make_shared<Eigen::Tensor<D, R>>(std::move(t))),
              gradFn(std::nullopt),
              requiresGrad(requiresGrad) {}

    explicit Tensor(const std::array<std::int64_t, R> &shape, bool requiresGrad = false)
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
    // compute the gradient of all predecessors w.r.t tensor
    void backward(D v = 1) {
        // neither this tensor nor any of its predecessors needs a gradient
        if (!gradFn.has_value())
            return;

        // initial gradient
        gradFn.value()->addGrad(data->constant(v));

        auto head = gradFn.value();

        // first go through the graph to count how many children each node has
        // so that each node only computes the gradients once
        std::queue<std::shared_ptr<CNodeBase>> q;
        q.push(head);
        while (!q.empty()) {
            auto e = q.front();
            q.pop();
            for (const auto& n : e->parents) {
                n->childrenThatNeedToComputeGradients++;
                if (!n->visited) {
                    n->visited = true;
                    q.push(n);
                }
            }
        }

        // backpropagate gradients
        q.push(head);
        while (!q.empty()) {
            auto e = q.front();
            q.pop();
            // if at least one children did not compute its gradients
            if (e->childrenThatNeedToComputeGradients > 0) {
                q.push(e);
                continue;
            }

            // compute gradients for parents
            e->computeGradients();
            for (const auto& n : e->parents) {
                n->childrenThatNeedToComputeGradients--;
                if (n->visited) {
                    n->visited = false;
                    q.push(n);
                }
            }
        }
    }
};





#endif //LIBDL_TENSOR_H
