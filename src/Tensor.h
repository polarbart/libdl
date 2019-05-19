//
// Created by polarbabe on 12.05.19.
//

#ifndef LIBDL_TENSOR_H
#define LIBDL_TENSOR_H

#include <memory>
#include <unsupported/Eigen/CXX11/Tensor>
#include "CNode.h"
#include <pybind11/numpy.h>

namespace py = pybind11;

template <typename D, int R>
class Tensor {
public:
    const py::array_t<D, py::array::f_style> data;
    const Eigen::TensorMap<Eigen::Tensor<D, R>> eTensor;
    bool requires_grad;

    explicit Tensor(pybind11::array_t<D, py::array::f_style>, bool = false);
    static std::shared_ptr<Tensor<D, R>> make_tensor(Eigen::TensorMap<Eigen::Tensor<D, R>>&);
    explicit Tensor(Eigen::TensorMap<Eigen::Tensor<D, R>>&);
    void setGradFn(const std::shared_ptr<CNode>&);
    std::optional<std::shared_ptr<CNode>> getGradFn();
    bool needsGradient();

private:

    std::optional<std::shared_ptr<CNode>> grad_fn;

    static Eigen::TensorMap<Eigen::Tensor<D, R>> intiTensorMap(py::buffer_info);
    static py::array_t<D, py::array::f_style> intiNumpyBuffer(Eigen::TensorMap<Eigen::Tensor<D, R>>&);
};





#endif //LIBDL_TENSOR_H
