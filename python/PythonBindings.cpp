
#include <utility>
#include <iostream>
#include <string>
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
#include "../src/ops/CNodeBase.h"

namespace py = pybind11;

template <typename D, std::int64_t R>
std::shared_ptr<Tensor<D, R>> constant(const std::array<std::int64_t, R> &shape, D value, bool requiresGrad) {
    auto ret = std::make_shared<Tensor<D, R>>(shape, requiresGrad);
    ret->data->setConstant(value);
    if (requiresGrad)
        ret->setGradFn(std::make_shared<Leaf<D, R>>(ret));
    return ret;
}

template <typename D, std::int64_t R>
std::shared_ptr<Tensor<D, R>> zeros(const std::array<std::int64_t, R> &shape, bool requiresGrad) {
    return constant<D, R>(shape, 0, requiresGrad);
}

template <typename D, std::int64_t R>
std::shared_ptr<Tensor<D, R>> ones(const std::array<std::int64_t, R> &shape, bool requiresGrad) {
    return constant<D, R>(shape, 1, requiresGrad);
}

template <typename D, std::int64_t R>
std::shared_ptr<Tensor<D, R>> uniform(const std::array<std::int64_t, R> &shape, D low, D high, bool requiresGrad) {
    auto ret = std::make_shared<Tensor<D, R>>(shape, requiresGrad);
    ret->data->device(GlobalThreadPool::myDevice) = ret->data->random() * (ret->data->constant(high) - ret->data->constant(low)) - ret->data->constant(low);
    if (requiresGrad)
        ret->setGradFn(std::make_shared<Leaf<D, R>>(ret));
    return ret;
}

template <typename D, std::int64_t R>
std::shared_ptr<Tensor<D, R>> centeredUniform(const std::array<std::int64_t, R> &shape, D lowhigh, bool requiresGrad) {
    return uniform<D, R>(shape, -lowhigh, lowhigh, requiresGrad);
}

template <typename D, std::int64_t R>
std::shared_ptr<Tensor<D, R>> normal(const std::array<std::int64_t, R> &shape, D mean, D std, bool requiresGrad) {
    static Eigen::internal::NormalRandomGenerator<D> rng;
    auto ret = std::make_shared<Tensor<D, R>>(shape, requiresGrad);
    ret->data->device(GlobalThreadPool::myDevice) = ret->data->random(rng) * ret->data->constant(std) + ret->data->constant(mean);
    if (requiresGrad)
        ret->setGradFn(std::make_shared<Leaf<D, R>>(ret));
    return ret;
}

template <typename D, std::int64_t R>
std::shared_ptr<Tensor<D, R>> fromNumpy(py::array_t<D, py::array::f_style> &array, bool requiresGrad) {
    auto info = array.request(true);
    std::array<std::int64_t, R> shape {};
    std::copy_n(std::begin(info.shape), R, std::begin(shape));

    Eigen::Tensor<D, R> data(shape);
    std::copy_n(static_cast<D*>(info.ptr), data.size(), data.data());

    auto ret = std::make_shared<Tensor<D, R>>(std::move(data), requiresGrad);
    if (requiresGrad)
        ret->setGradFn(std::make_shared<Leaf<D, R>>(ret));
    return ret;
}


template<typename D, std::int64_t RA, std::int64_t RB>
void init_datatype_dimension_dimesnion(py::module &m, py::class_<Tensor<D, RA>, std::shared_ptr<Tensor<D, RA>>> tensor) {
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



template<typename D, std::int64_t R>
void init_datatpye_dimension(py::module &m) {

    std::string eigenTensorName = "_EigenTensor" + std::to_string(R);
    auto eigenTensor = py::class_<Eigen::Tensor<D, R>, std::shared_ptr<Eigen::Tensor<D, R>>>(m, eigenTensorName.c_str(), py::buffer_protocol())
            .def_buffer([](Eigen::Tensor<D, R> &t) -> py::buffer_info {
                std::array<ssize_t, R> strides {};
                if constexpr (R > 0) {
                    strides[0] = sizeof(D);
                    for (std::int64_t i = 1; i < R; i++)
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
            })
            .def("__str__", [](Eigen::Tensor<D, R> &t) {
                std::string ret;
                if constexpr (R > 1) {
                    ret = "only supported for tensors of dimension 1 or 0";
                } else {
                    std::stringstream ss;
                    ss << "[ ";
                    for (std::int64_t i = 0; i < t.size(); i++) {
                        ss << t.data()[i] << " ";
                    }
                    ss << "]";
                    ret = ss.str();
                }
                return ret;
            });

    if constexpr (R >= 1)
        eigenTensor.def("__getitem__", [](Eigen::Tensor<D, R> &t, std::array<std::int64_t, R> idx) {
            for (std::int64_t i = 0; i < R; i++)
                if (idx[i] < 0 || idx[i] >= t.dimension(i))
                    throw std::invalid_argument("index out of range");
            return t(idx);
        });
    if constexpr (R <= 1)
        eigenTensor.def("__getitem__", [](Eigen::Tensor<D, R> &t, std::int64_t i) {
            if ((R == 0 && i != 0) || (R == 1 && (i < 0 || i >= t.dimension(0))))
                throw std::invalid_argument("index out of range");
            return t(i);
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
                                  if (!t->gradFn.has_value())
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
            .def_property_readonly("shape", [](const Tensor<D, R> &t){return static_cast<std::array<std::int64_t, R>>(t.data->dimensions());})
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
    m.def("zeros", &zeros<D, R>);
    m.def("ones", &ones<D, R>);
    m.def("uniform", &uniform<D, R>);
    m.def("uniform", &centeredUniform<D, R>);
    m.def("normal", &normal<D, R>);
    m.def("sigmoid", &Sigmoid<D, R>::sigmoid);
    m.def("leaky_relu", &LeakyRelu<D, R>::leakyRelu, py::arg("x"), py::arg("negativeSlope") = 0.01);
    m.def("relu", &Relu<D, R>::relu);
    m.def("pow", &Pow<D, R>::pow);
    m.def("apply_adam", &Adam<D, R>::applyAdam);
    if constexpr (R > 0) {
        m.def("sum", &Sum<D, R>::sum);
        m.def("mean", &Mean<D, R>::mean);
    }
    init_datatype_dimension_dimesnion<D, R, 0>(m, tensor);
            init_datatype_dimension_dimesnion<D, R, 1>(m, tensor);
    init_datatype_dimension_dimesnion<D, R, 2>(m, tensor);
    init_datatype_dimension_dimesnion<D, R, 3>(m, tensor);
    init_datatype_dimension_dimesnion<D, R, 4>(m, tensor);
}

template <typename D>
void init_datatpye(py::module &m) {

    m.def("conv_2d", &Conv2D<D>::conv2d);
    m.def("maxpool_2d", &MaxPool2D<D>::maxpool2d);
    m.def("batch_norm_2d", &BatchNorm2D<D>::batchNorm2d);
    m.def("linear", &Linear<D>::linear);
    m.def("cross_entropy_with_logits", &CrossEntropyWithLogits<D>::crossEntropyWithLogits);

    init_datatpye_dimension<D, 0>(m);
    init_datatpye_dimension<D, 1>(m);
    init_datatpye_dimension<D, 2>(m);
    init_datatpye_dimension<D, 3>(m);
    init_datatpye_dimension<D, 4>(m);
}

PYBIND11_MODULE(libdl_python, m) {
    m.def("set_no_grad", [](bool nograd) {CNodeBase::noGrad = nograd;});
    m.def("get_no_grad", []() {return CNodeBase::noGrad;});
    init_datatpye<std::float_t>(m);
}