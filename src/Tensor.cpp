#include <utility>

//
// Created by polarbabe on 12.05.19.
//

#include "Tensor.h"

Tensor::Tensor(pybind11::array_t<float, py::array::f_style> data) : data(std::move(data)) {
    bufferInfo = data.request(true);
}

Eigen::TensorMap<Eigen::Tensor<float, 4>> Tensor::getAsEigen() {
    switch (bufferInfo.ndim) {
        case 1:
            return Eigen::TensorMap<Eigen::Tensor<float, 4>> (static_cast<float*>(bufferInfo.ptr), 1, 1, 1, bufferInfo.shape[0]);
        case 2:
            return Eigen::TensorMap<Eigen::Tensor<float, 4>> (static_cast<float*>(bufferInfo.ptr), 1, 1, bufferInfo.shape[0], bufferInfo.shape[1]);
        case 3:
            return Eigen::TensorMap<Eigen::Tensor<float, 4>> (static_cast<float*>(bufferInfo.ptr), 1, bufferInfo.shape[0], bufferInfo.shape[1], bufferInfo.shape[2]);
        case 4:
            return Eigen::TensorMap<Eigen::Tensor<float, 4>> (static_cast<float*>(bufferInfo.ptr), bufferInfo.shape[0], bufferInfo.shape[1], bufferInfo.shape[2], bufferInfo.shape[3]);
        default:
            throw "Nana TODO";
    }
}

void Tensor::setFromEigen(const Eigen::Tensor<float, 4> &) {

}

template<int R>
Tensor::Tensor(Eigen::Tensor<float, R>) {

}

template<int R>
Eigen::TensorMap<Eigen::Tensor<float, R>> Tensor::getEigen() {
    return Eigen::TensorMap<Eigen::Tensor<float, R>>(nullptr);
}


PYBIND11_MODULE(libdl, m) {
    py::class_<Tensor>(m, "Tensor")
            .def(py::init<py::array_t<float, py::array::f_style>>())
            .def_readwrite("data", &Tensor::data)
            .def_readwrite("requires_grad", &Tensor::requires_grad);
}