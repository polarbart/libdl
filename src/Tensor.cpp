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
}

template<typename D, int R>
void init_Module1(py::module &m) {
    std::string name = "Tensor" + std::to_string(R);
    auto tensor = py::class_<Tensor<D, R>, std::shared_ptr<Tensor<D, R>>>(m, name.c_str())
            .def(py::init(&Tensor<D, R>::fromNumpy))
            .def("numpy", [](const Tensor<D, R> &t) {return std::static_pointer_cast<ETensor<D, R>>(t.eTensor)->array;})
            .def_readwrite("requires_grad", &Tensor<D, R>::requiresGrad)
            .def("backward", &Tensor<D, R>::backward, py::arg("v") = 1)
            .def("grad", [](const Tensor<D, R> &t) {return std::static_pointer_cast<ETensor<D, R>>(t.grad)->array;})
            .def("zero_grad", &Tensor<D, R>::zeroGrad)
            .def("apply_gradient", &Tensor<D, R>::applyGradient)
            .def("__pow__", &Pow<D, R>::pow)
            .def_property_readonly("shape", [](const Tensor<D, R> &t){return static_cast<std::array<long, R>>(t.eTensor->dimensions());});
    m.def("sigmoid", &Sigmoid<D, R>::sigmoid);
    m.def("leaky_relu", &LeakyRelu<D, R>::leakyRelu);
    m.def("relu", &Relu<D, R>::relu);
    m.def("sigmoid", &Sigmoid<D, R>::sigmoid);
    m.def("pow", &Pow<D, R>::pow);
    m.def("apply_adam", &Adam<D, R>::applyAdam);
    if constexpr (R > 0) {
        m.def("sum", &SumAlongAxis<D, R>::sum);
        m.def("mean", &MeanAlongAxis<D, R>::mean);
        m.def("sum", &Sum<D, R>::sum);
        m.def("mean", &Mean<D, R>::mean);
        m.def("cross_entropy_with_logits", &CrossEntropyWithLogits<D, R>::crossEntropyWithLogits);
    }
    if constexpr (R == 0) {
        m.def("conv_2d", &Conv2D<D>::conv2d);
        m.def("maxpool_2d", &MaxPool2D<D>::maxpool2d);
        m.def("batch_norm_2d", &BatchNorm2D<D>::batchNorm2d);
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
}