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
    py::array_t<D, py::array::f_style> data;
    Eigen::TensorMap<Eigen::Tensor<D, R>> eTensor;
    bool requires_grad;

    Tensor(std::shared_ptr<D[]>, std::array<long, R>&, bool = false);

    static Tensor<D, R>* fromNumpy(py::array_t<D, py::array::f_style>, bool = false);

    void setGradFn(const std::shared_ptr<CNode>&);
    std::optional<std::shared_ptr<CNode>> getGradFn();
    bool needsGradient();

private:

    std::optional<std::shared_ptr<CNode>> grad_fn;
    std::shared_ptr<D[]> iData;

    static Eigen::TensorMap<Eigen::Tensor<D, R>> intiTensorMap(py::buffer_info);
    static py::array_t<D, py::array::f_style> initNpArray(std::shared_ptr<D[]>, std::array<long, R>&);
};





#endif //LIBDL_TENSOR_H
