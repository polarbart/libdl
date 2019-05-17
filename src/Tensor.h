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

    Tensor(pybind11::array_t<D, py::array::f_style>, bool = false);
    explicit Tensor(const Eigen::Tensor<D, R>&);
    void setGradFn(std::shared_ptr<CNode>);
    std::optional<std::shared_ptr<CNode>> getGradFn();
    bool needsGradient();

protected:

private:
    const py::buffer_info bufferInfo;
    std::optional<std::shared_ptr<CNode>> grad_fn;

    const Eigen::TensorMap<Eigen::Tensor<D, R>> intiTensorMap();
    const py::array_t<D, py::array::f_style> intiNumpyBuffer();
};



#endif //LIBDL_TENSOR_H
