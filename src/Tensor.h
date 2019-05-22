//
// Created by polarbabe on 12.05.19.
//

#ifndef LIBDL_TENSOR_H
#define LIBDL_TENSOR_H

#include <iostream>
#include <memory>
#include <unsupported/Eigen/CXX11/Tensor>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <queue>
#include "CNodeBase.h"

template<typename D, int R>
class CNode;

template<typename D, int R>
class Leaf;

namespace py = pybind11;

template <typename D, int R>
class Tensor {
public:
    py::array_t<D, py::array::f_style> data;
    Eigen::TensorMap<Eigen::Tensor<D, R>> eTensor;
    bool requires_grad;

    Tensor(std::shared_ptr<D[]> d, const std::array<long, R>& shape, bool requiresGrad = false)
            : data(initNpArray(d.get(), shape)),
              eTensor(Eigen::TensorMap<Eigen::Tensor<D, R>>(d.get(), shape)),
              gradFn(std::nullopt),
              requires_grad(requiresGrad),
              iData(d) {}

    static std::shared_ptr<Tensor<D, R>> fromNumpy(py::array_t<D, py::array::f_style> array, bool requiresGrad = false) {
        auto info = array.request(false);
        auto d = std::shared_ptr<D[]>(new D[info.size]);
        std::copy_n(static_cast<D*>(info.ptr), info.size, d.get());
        std::array<long, R> shape {};
        for (int i = 0; i < R; ++i)
            shape[i] = info.shape[i];
        auto ret = std::shared_ptr<Tensor<D, R>>(new Tensor<D, R>(d, shape, requiresGrad));
        ret->setGradFn(std::make_shared<Leaf<D, R>>(ret));
        return ret;
    }

    void setGradFn(const std::shared_ptr<CNode<D, R>>& g) {
        gradFn = std::optional<std::shared_ptr<CNode<D, R>>>(g);
    }

    std::optional<std::shared_ptr<CNode<D, R>>> gradFn;

    bool needsGradient() {
        return requires_grad || gradFn.has_value();
    }

    void addGrad(const Eigen::Tensor<D, R>& g) {
        if (resetGrad) {
            grad = g;
            resetGrad = false;
        } else
            grad += g;
    }

    void zeroGrad() {
        resetGrad = true;
    }

    void backward(float v = 1) {
        if (!gradFn.has_value())
            std::cout << "no grad is computed" << std::endl;
        Eigen::Tensor<D, R> g = eTensor.constant(v);
        // std::cout << gradFn.value()->grad << std::endl;
        gradFn.value()->addGrad(g);

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

    void applyGradient(D lr) {
        eTensor -= grad.constant(lr) + grad;
    }

    py::array_t<D, py::array::f_style> npgrad() {
        if constexpr (R == 0)
            return py::array_t<D, py::array::f_style>(std::vector<size_t> {1}, std::array<size_t, 1> {sizeof(D)}, grad.data());
        else {
            size_t strides[R];
            std::vector<long> shape {grad.dimension(0)};
            strides[0] = sizeof(D);
            for (int i = 1; i < R; ++i) {
                strides[i] = grad.dimension(i) * strides[i - 1];
                shape.push_back(grad.dimension(i));
            }
            return py::array_t<D, py::array::f_style>(shape, strides, grad.data());
        }
    }

private:
    Eigen::Tensor<D, R> grad;
    bool resetGrad = true;
    std::shared_ptr<D[]> iData;

    static py::array_t<D, py::array::f_style> initNpArray(D* d, const std::array<long, R>& shape) {
        if constexpr (R == 0)
            return py::array_t<D, py::array::f_style>(std::vector<size_t> {1}, std::array<size_t, 1> {sizeof(D)}, d);
        else {
            size_t strides[R];
            strides[0] = sizeof(D);
            for (int i = 1; i < R; ++i)
                strides[i] = shape[i] * strides[i - 1]; // TODO right?
            return py::array_t<D, py::array::f_style>(shape, strides, d);
        }
    }
};





#endif //LIBDL_TENSOR_H
