
#ifndef LIBDL_TENSOR_H
#define LIBDL_TENSOR_H
#define EIGEN_USE_THREADS

#include <iostream>
#include <memory>
#include <unsupported/Eigen/CXX11/Tensor>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <queue>
#include "ops/CNode.h"
#include "ETensor.h"
#include "ops/Leaf.h"


namespace py = pybind11;

template <typename D, int R>
class Tensor {
public:

    std::shared_ptr<Eigen::TensorMap<Eigen::Tensor<D, R>>> eTensor;
    std::shared_ptr<Eigen::TensorMap<Eigen::Tensor<D, R>>> grad;
    std::optional<std::shared_ptr<CNode<D, R>>> gradFn;
    bool requiresGrad;

    template<typename OtherDerived>
    Tensor(const OtherDerived &t, const std::array<long, R> &shape)
            : eTensor(std::make_shared<ETensor<D, R>>(t, shape)),
              gradFn(std::nullopt),
              requiresGrad(false) {}

    explicit Tensor(const py::array_t<D, py::array::f_style> &array, bool requiresGrad = false)
            : eTensor(std::make_shared<ETensor<D, R>>(array)),
              gradFn(std::nullopt),
              requiresGrad(requiresGrad) {}

    explicit Tensor(const std::array<long, R> &shape, bool requiresGrad = false)
            : eTensor(std::make_shared<ETensor<D, R>>(shape)),
              gradFn(std::nullopt), // TODO leaf node
              requiresGrad(requiresGrad) {}

    static std::shared_ptr<Tensor<D, R>> fromNumpy(const py::array_t<D, py::array::f_style> &array, bool requiresGrad = false) {
        auto t = std::make_shared<Tensor<D, R>>(array, requiresGrad);
        if (requiresGrad)
            t->setGradFn(std::make_shared<Leaf<D, R>>(t));
        return t;
    }

    void setGradFn(const std::shared_ptr<CNode<D, R>>& g) {
        gradFn = std::optional<std::shared_ptr<CNode<D, R>>>(g);
    }

    bool needsGradient() {
        return requiresGrad || gradFn.has_value();
    }

    void zeroGrad() {
        grad = std::shared_ptr<ETensor<D, R>>(nullptr);
    }

    void subGrad(D lr) {
        static Eigen::ThreadPool pool(8);
        static Eigen::ThreadPoolDevice myDevice(&pool, 8);
        if (grad.use_count() > 0)
            eTensor->device(myDevice) -= grad->constant(lr) * *grad;
    }

    void addGrad(const std::shared_ptr<Eigen::TensorMap<Eigen::Tensor<D, R>>> &g) {
        static Eigen::ThreadPool pool(8);
        static Eigen::ThreadPoolDevice myDevice(&pool, 8);
        if (grad.use_count() == 0) {
            grad = g;
        } else if (grad != g)
            grad->device(myDevice) += *g;
    }

    void backward(D v = 1) {
        if (!gradFn.has_value()) {
            std::cout << "no grad is computed" << std::endl;
            return;
        }
        gradFn.value()->addGrad(eTensor->constant(v));

        auto head = gradFn.value();

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
