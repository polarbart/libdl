//
// Created by polarbabe on 26.05.19.
//

#ifndef LIBDL_ETENSOR_H
#define LIBDL_ETENSOR_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <pybind11/numpy.h>

namespace py = pybind11;

template <typename D, int R>
class ETensor : public Eigen::TensorMap<Eigen::Tensor<D, R>> {

public:

    explicit ETensor(const py::array_t<D, py::array::f_style> a) : Eigen::TensorMap<Eigen::Tensor<D, R>>(toTensorMap(a)), array(a) {}

    explicit ETensor(const std::array<long, R> &shape)
        : Eigen::TensorMap<Eigen::Tensor<D, R>>(new D[std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>())], shape) {
        py::capsule c(Eigen::TensorMap<Eigen::Tensor<D, R>>::data(), [](void *f) {
            D *foo = reinterpret_cast<D*>(f);
            delete[] foo;
        });
        array = py::array_t<D, py::array::f_style>(shape, Eigen::TensorMap<Eigen::Tensor<D, R>>::data(), c);
    }

    template <typename OtherDerived>
    explicit ETensor(const OtherDerived &t, const std::array<long, R> &d)
        : Eigen::TensorMap<Eigen::Tensor<D, R>>(new D[std::accumulate(std::begin(d), std::end(d), 1, std::multiplies<>())], d){
        Eigen::TensorMap<Eigen::Tensor<D, R>>::operator=(t);
        py::capsule c(Eigen::TensorMap<Eigen::Tensor<D, R>>::data(), [](void *f) {
            D *foo = reinterpret_cast<D*>(f);
            delete[] foo;
        });
        std::array<long, R> shape {};
        std::copy_n(std::begin(Eigen::TensorMap<Eigen::Tensor<D, R>>::dimensions()), R, std::begin(shape));
        array = py::array_t<D, py::array::f_style>(shape, Eigen::TensorMap<Eigen::Tensor<D, R>>::data(), c);
    }

    py::array_t<D, py::array::f_style> array;

private:
    static Eigen::TensorMap<Eigen::Tensor<D, R>> toTensorMap(py::array_t<D, py::array::f_style> a) {
        auto info = a.request(true);
        std::array<long, R> shape;
        for (int i = 0; i < R; ++i)
            shape[i] = info.shape[i];
        return Eigen::TensorMap<Eigen::Tensor<D, R>>(static_cast<D*>(info.ptr), shape);
    }
};


#endif //LIBDL_ETENSOR_H
