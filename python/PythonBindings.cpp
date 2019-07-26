
#include <utility>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "../src/Tensor.h"
#include "../src/ops/Add.h"
#include "../src/ops/MatMul.h"
#include "../src/ops/Leaf.h"
#include "../src/ops/Sub.h"
#include "../src/ops/Sigmoid.h"
#include "../src/ops/Pow.h"
#include "../src/ops/SumAlongAxes.h"
#include "../src/ops/MeanAlongAxes.h"
#include "../src/ops/Sum.h"
#include "../src/ops/Mean.h"
#include "../src/ops/LeakyRelu.h"
#include "../src/ops/Relu.h"
#include "../src/ops/Conv2D.h"
#include "../src/ops/MaxPool2D.h"
#include "../src/ops/Reshape.h"
#include "../src/ops/CrossEntropyWithLogits.h"
#include "../src/ops/BatchNorm2D.h"
#include "../src/functional/Adam.h"
#include "../src/ops/Linear.h"

namespace py = pybind11;

template <typename D, int R>
std::shared_ptr<Tensor<D, R>> constant(std::array<long, R> shape, D value, bool requiresGrad) {
    auto ret = std::make_shared<Tensor<D, R>>(shape, requiresGrad);
    ret->data->setConstant(value);
    if (requiresGrad)
        ret->setGradFn(std::make_shared<Leaf<D, R>>(ret));
    return ret;
}

template <typename D, int R>
std::shared_ptr<Tensor<D, R>> zeros(std::array<long, R> shape, bool requiresGrad) {
    return constant<D, R>(shape, 0, requiresGrad);
}

template <typename D, int R>
std::shared_ptr<Tensor<D, R>> ones(std::array<long, R> shape, bool requiresGrad) {
    return constant<D, R>(shape, 1, requiresGrad);
}

template <typename D, int R>
std::shared_ptr<Tensor<D, R>> uniform(std::array<long, R> shape, D low, D high, bool requiresGrad) {
    auto ret = std::make_shared<Tensor<D, R>>(shape, requiresGrad);
    static Eigen::ThreadPool pool(8);
    static Eigen::ThreadPoolDevice myDevice(&pool, 8);
    ret->data->device(myDevice) = ret->data->random() * (ret->data->constant(high) - ret->data->constant(low)) - ret->data->constant(low);
    if (requiresGrad)
        ret->setGradFn(std::make_shared<Leaf<D, R>>(ret));
    return ret;
}

template <typename D, int R>
std::shared_ptr<Tensor<D, R>> centeredUniform(std::array<long, R> shape, D lowhigh, bool requiresGrad) {
    return uniform<D, R>(shape, lowhigh, lowhigh, requiresGrad);
}

template <typename D, int R>
std::shared_ptr<Tensor<D, R>> fromNumpy(py::array_t<D, py::array::f_style> &array, bool requiresGrad) {
    auto info = array.request(true);
    std::array<long, R> shape {};
    std::copy_n(std::begin(info.shape), R, std::begin(shape));

    Eigen::Tensor<D, R> data(shape);
    std::copy_n(static_cast<D*>(info.ptr), data.size(), data.data());

    auto ret = std::make_shared<Tensor<D, R>>(std::move(data), requiresGrad);
    if (requiresGrad)
        ret->setGradFn(std::make_shared<Leaf<D, R>>(ret));
    return ret;
}


template<typename D, int RA, int RB>
void init_Module2(py::module &m, py::class_<Tensor<D, RA>, std::shared_ptr<Tensor<D, RA>>> tensor) {
    m.def("add", &Add<D, RA, RB>::add);
    tensor.def("__add__", &Add<D, RA, RB>::add);
    m.def("sub", &Sub<D, RA, RB>::sub);
    tensor.def("__sub__", &Sub<D, RA, RB>::sub);
    if constexpr (RA > 0 && RB > 0 && (RA + RB - 2) > 0 && (RA + RB - 2) <= 5)
        m.def("matmul", &MatMul<D, RA, RB>::matmul);
    if constexpr (RB > 0)
        m.def("reshape", &Reshape<D, RA, RB>::reshape);
    if constexpr (RA >= RB && RB > 0) {
        m.def("sum", &SumAlongAxes<D, RA, RB>::sum);
        m.def("mean", &MeanAlongAxes<D, RA, RB>::mean);
    }
}



template<typename D, int R>
void init_Module1(py::module &m) {

    std::string eigenTensorName = "_EigenTensorx" + std::to_string(R);
    py::class_<Eigen::Tensor<D, R>, std::shared_ptr<Eigen::Tensor<D, R>>>(m, eigenTensorName.c_str(), py::buffer_protocol())
            .def_buffer([](Eigen::Tensor<D, R> &t) -> py::buffer_info {
                std::array<ssize_t, R> strides {};
                if constexpr (R > 0) {
                    strides[0] = sizeof(D);
                    for (int i = 1; i < R; i++)
                        strides[i] = t.dimension(i - 1) * strides[i - 1];
                }
                return py::buffer_info(
                        t.data(),
                        sizeof(D),
                        py::format_descriptor<D>::format(),
                        R,
                        t.dimensions(),
                        strides
                );
            });


    std::string myTensorName = "Tensor" + std::to_string(R);
    auto tensor = py::class_<Tensor<D, R>, std::shared_ptr<Tensor<D, R>>>(m, myTensorName.c_str())
            .def(py::init(&fromNumpy<D, R>))
            .def_readonly("data", &Tensor<D, R>::data)
            .def_readonly("grad", &Tensor<D, R>::grad)
            .def_property("requires_grad", [](const std::shared_ptr<Tensor<D, R>> &t) {return t->requiresGrad;},
                          [](std::shared_ptr<Tensor<D, R>> t, bool requiresGrad) {
                              if (requiresGrad) {
                                  t->requiresGrad = true;
                                  t->setGradFn(std::make_shared<Leaf<D, R>>(t));
                              } else {
                                  t->requiresGrad = false;
                                  t->gradFn = std::nullopt;
                              }
                          })
            .def("backward", &Tensor<D, R>::backward, py::arg("v") = 1)
            .def("zero_grad", &Tensor<D, R>::zeroGrad)
            .def("sub_grad", &Tensor<D, R>::subGrad)
            .def("__pow__", &Pow<D, R>::pow)
            .def_property_readonly("shape", [](const Tensor<D, R> &t){return static_cast<std::array<long, R>>(t.data->dimensions());})
            .def(py::pickle(
                    [](const Tensor<D, R> &t) {
                        return py::make_tuple(py::array_t<D, py::array::f_style>(t.data->dimensions(), t.data->data()), t.requiresGrad);
                    },
                    [](py::tuple t) {
                        auto array = t[0].cast<py::array_t<D, py::array::f_style>>();
                        return fromNumpy<D, R>(array, t[1].cast<bool>());
                    }
                ));
    m.def("constant", &constant<D, R>);
    m.def("uniform", &uniform<D, R>);
    m.def("uniform", &centeredUniform<D, R>);
    m.def("zeros", &zeros<D, R>);
    m.def("ones", &ones<D, R>);
    m.def("sigmoid", &Sigmoid<D, R>::sigmoid);
    m.def("leaky_relu", &LeakyRelu<D, R>::leakyRelu, py::arg("x"), py::arg("negativeSlope") = 0.01);
    m.def("relu", &Relu<D, R>::relu);
    m.def("sigmoid", &Sigmoid<D, R>::sigmoid);
    m.def("pow", &Pow<D, R>::pow);
    m.def("apply_adam", &Adam<D, R>::applyAdam);
    if constexpr (R > 0) {
        m.def("sum", &Sum<D, R>::sum);
        m.def("mean", &Mean<D, R>::mean);
    }
    if constexpr (R == 0) {
        m.def("conv_2d", &Conv2D<D>::conv2d);
        m.def("maxpool_2d", &MaxPool2D<D>::maxpool2d);
        m.def("batch_norm_2d", &BatchNorm2D<D>::batchNorm2d);
        m.def("linear", &Linear<D>::linear);
        m.def("cross_entropy_with_logits", &CrossEntropyWithLogits<D>::crossEntropyWithLogits);
    }
    init_Module2<D, R, 0>(m, tensor);
    init_Module2<D, R, 1>(m, tensor);
    init_Module2<D, R, 2>(m, tensor);
    init_Module2<D, R, 3>(m, tensor);
    init_Module2<D, R, 4>(m, tensor);
    //init_Module2<D, R, 5>(m);
}

PYBIND11_MODULE(libdl_python, m) {
    init_Module1<float, 0>(m);
    init_Module1<float, 1>(m);
    init_Module1<float, 2>(m);
    init_Module1<float, 3>(m);
    init_Module1<float, 4>(m);
    //init_Module1<float, 5>(m);
}