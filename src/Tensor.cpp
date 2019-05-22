#include <utility>
#include <iostream>

#include "Tensor.h"
#include "MatMul.h"
#include "Add.h"
#include "Leaf.h"
#include "Sub.h"
#include "Sigmoid.h"
#include "Pow.h"
#include "SumAlongAxis.h"
#include "MeanAlongAxis.h"
#include "Sum.h"
#include "Mean.h"

template<typename D, int RA, int RB>
void init_Module2(py::module &m, py::class_<Tensor<D, RA>, std::shared_ptr<Tensor<D, RA>>> tensor) {
    m.def("add", &Add<D, RA, RB>::add);
    tensor.def("__add__", &Add<D, RA, RB>::add);
    m.def("sub", &Sub<D, RA, RB>::sub);
    tensor.def("__sub__", &Sub<D, RA, RB>::sub);
    if constexpr (RA > 0 && RB > 0 && (RA + RB - 2) > 0 && (RA + RB - 2) <= 5)
        m.def("matmul", &MatMul<D, RA, RB>::matmul);
}

template<typename D, int R>
void init_Module1(py::module &m) {
    std::string name = "Tensor" + std::to_string(R);
    auto tensor = py::class_<Tensor<D, R>, std::shared_ptr<Tensor<D, R>>>(m, name.c_str())
            .def(py::init(&Tensor<D, R>::fromNumpy))
            .def_readwrite("data", &Tensor<D, R>::data)
            .def_readwrite("requires_grad", &Tensor<D, R>::requires_grad)
            .def("needsGradient", &Tensor<D, R>::needsGradient)
            .def("backward", &Tensor<D, R>::backward, py::arg("v") = 1)
            .def("npgrad", &Tensor<D, R>::npgrad)
            .def("zeroGrad", &Tensor<D, R>::zeroGrad)
            .def("applyGradient", &Tensor<D, R>::applyGradient)
            .def("__pow__", &Pow<D, R>::pow);
    m.def("sigmoid", &Sigmoid<D, R>::sigmoid);
    m.def("pow", &Pow<D, R>::pow);
    if constexpr (R > 0) {
        m.def("sum", &SumAlongAxis<D, R>::sum);
        m.def("mean", &MeanAlongAxis<D, R>::mean);
        m.def("sum", &Sum<D, R>::sum);
        m.def("mean", &Mean<D, R>::mean);
    }
    init_Module2<D, R, 0>(m, tensor);
    init_Module2<D, R, 1>(m, tensor);
    init_Module2<D, R, 2>(m, tensor);
    //init_Module2<D, R, 3>(m);
    //init_Module2<D, R, 4>(m);
    //init_Module2<D, R, 5>(m);
}

PYBIND11_MODULE(libdl, m) {
    init_Module1<float, 0>(m);
    init_Module1<float, 1>(m);
    init_Module1<float, 2>(m);
    //init_Module1<float, 3>(m);
    //init_Module1<float, 4>(m);
    //init_Module1<float, 5>(m);
}