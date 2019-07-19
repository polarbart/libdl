#include <utility>
#include <iostream>

#include "Tensor.h"
#include "Add.h"
#include "MatMul.h"
#include "Leaf.h"
#include "Sub.h"
#include "Sigmoid.h"
#include "Pow.h"
#include "SumAlongAxis.h"
#include "MeanAlongAxis.h"
#include "Sum.h"
#include "Mean.h"
#include "LeakyRelu.h"
#include "Relu.h"
#include "Conv2D.h"
#include "MaxPool2D.h"
#include "Reshape.h"
#include "CrossEntropyWithLogits.h"
#include "BatchNorm2D.h"
#include "Adam.h"
#include "Linear.h"

template <typename D, int R>
std::shared_ptr<Tensor<D, R>> constant(std::array<long, R> shape, D value, bool requires_grad) {
    auto ret = std::make_shared<Tensor<D, R>>(shape, requires_grad);
    ret->eTensor->setConstant(value);
    return ret;
}

template <typename D, int R>
std::shared_ptr<Tensor<D, R>> uniform(std::array<long, R> shape, D low, D high, bool requires_grad) {
    auto ret = std::make_shared<Tensor<D, R>>(shape, requires_grad);
    static Eigen::ThreadPool pool(8);
    static Eigen::ThreadPoolDevice myDevice(&pool, 8);
    ret->eTensor->device(myDevice) = ret->eTensor->random() * (ret->eTensor->constant(high) - ret->eTensor->constant(low)) - ret->eTensor->constant(low);
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
        //m.def("sum", &SumAlongAxis<D, RA, RB>::sum);
        m.def("mean", &MeanAlongAxis<D, RA, RB>::mean);
    }
}

template<typename D, int R>
void init_Module1(py::module &m) {
    std::string name = "Tensor" + std::to_string(R);
    auto tensor = py::class_<Tensor<D, R>, std::shared_ptr<Tensor<D, R>>>(m, name.c_str())
            .def(py::init(&Tensor<D, R>::fromNumpy))
            .def("numpy", [](const Tensor<D, R> &t) {return std::static_pointer_cast<ETensor<D, R>>(t.eTensor)->array;})
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
            .def("grad", [](const Tensor<D, R> &t) {return std::static_pointer_cast<ETensor<D, R>>(t.grad)->array;})
            .def("zero_grad", &Tensor<D, R>::zeroGrad)
            .def("apply_gradient", &Tensor<D, R>::applyGradient)
            .def("__pow__", &Pow<D, R>::pow)
            .def_property_readonly("shape", [](const Tensor<D, R> &t){return static_cast<std::array<long, R>>(t.eTensor->dimensions());})
            .def("__getstate__", [](const Tensor<D, R> &p) {
                return py::make_tuple(std::static_pointer_cast<ETensor<D, R>>(p.eTensor)->array, p.requiresGrad);
            })
            .def("__setstate__", [](Tensor<D, R> &p, py::tuple t) {
                new (&p) Tensor<D, R>(t[0].cast<py::array_t<D, py::array::f_style>>(), t[1].cast<bool>());
            })
            .def("set_grad_fn()", [](const std::shared_ptr<Tensor<D, R>> &t) {
                if (t->requiresGrad)
                    t->setGradFn(std::make_shared<Leaf<D, R>>(t));
            });
    m.def("constant", &constant<D, R>);
    m.def("uniform", &uniform<D, R>);
    m.def("sigmoid", &Sigmoid<D, R>::sigmoid);
    m.def("leaky_relu", &LeakyRelu<D, R>::leakyRelu, py::arg("x"), py::arg("negativeSlope") = 0.01);
    m.def("relu", &Relu<D, R>::relu);
    m.def("sigmoid", &Sigmoid<D, R>::sigmoid);
    m.def("pow", &Pow<D, R>::pow);
    m.def("apply_adam", &Adam<D, R>::applyAdam);
    if constexpr (R > 0) {
        m.def("sum", &Sum<D, R>::sum);
        m.def("mean", &Mean<D, R>::mean);
        m.def("cross_entropy_with_logits", &CrossEntropyWithLogits<D, R>::crossEntropyWithLogits);
    }
    if constexpr (R == 0) {
        m.def("conv_2d", &Conv2D<D>::conv2d);
        m.def("maxpool_2d", &MaxPool2D<D>::maxpool2d);
        m.def("batch_norm_2d", &BatchNorm2D<D>::batchNorm2d);
        m.def("linear", &Linear<D>::linear);
    }
    init_Module2<D, R, 0>(m, tensor);
    init_Module2<D, R, 1>(m, tensor);
    init_Module2<D, R, 2>(m, tensor);
    init_Module2<D, R, 3>(m, tensor);
    init_Module2<D, R, 4>(m, tensor);
    //init_Module2<D, R, 5>(m);
}

PYBIND11_MODULE(libdl, m) {
    init_Module1<float, 0>(m);
    init_Module1<float, 1>(m);
    init_Module1<float, 2>(m);
    init_Module1<float, 3>(m);
    init_Module1<float, 4>(m);
    //init_Module1<float, 5>(m);

    /*py::enum_<PaddingType>(m, "PaddingType")
            .value("SAME", PaddingType::SAME)
            .value("VALID", PaddingType::VALID)
            .export_values();
    */
}