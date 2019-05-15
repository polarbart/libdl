//
// Created by polarbabe on 12.05.19.
//

#ifndef LIBDL_TENSOR_H
#define LIBDL_TENSOR_H

#include <memory>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CNode.h"
#include <pybind11/numpy.h>

#define MAKEALL(a, )

namespace py = pybind11;

class Tensor {
public:
    py::array_t<float, py::array::f_style> data;
    bool requires_grad = false;

    Tensor(pybind11::array_t<float, py::array::f_style>);

    template <int R>
    Tensor(Eigen::Tensor<float, R>);

    // void backward();

    template <int R>
    Eigen::TensorMap<Eigen::Tensor<float, R>> getEigen();

protected:

private:
    py::buffer_info bufferInfo;
    std::optional<std::shared_ptr<CNode>> grad_fn;
    Eigen::TensorMap<Eigen::Tensor<float, 1>> e1;
    Eigen::TensorMap<Eigen::Tensor<float, 2>> e2;
    Eigen::TensorMap<Eigen::Tensor<float, 3>> e3;
    Eigen::TensorMap<Eigen::Tensor<float, 4>> e4;

};




#endif //LIBDL_TENSOR_H
