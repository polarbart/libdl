//
// Created by polarbabe on 20.05.19.
//

#include "TestTensor.h"
#include "Add.h"

std::shared_ptr<TestTensor> TestTensor::add(const std::shared_ptr<TestTensor>& b) {
    if (rank != b->rank)
        throw "NANANA";
    std::shared_ptr<TestTensor> t;
    switch (rank) {
        case 1:
            t = std::make_shared<TestTensor>(t1 + b->t1);
            break;
        case 2:
            t =  std::make_shared<TestTensor>(t2 + b->t2);
            break;
        case 3:
            t = std::make_shared<TestTensor>(t3 + b->t3);
            break;
        case 4:
            t = std::make_shared<TestTensor>(t4 + b->t4);
            break;
        default:
            throw "NANANA";
    }
    if (needsGradient() || b->needsGradient())
        t->setGradFn(std::make_shared<Add>(grad_fn, b->grad_fn, t));
}


std::shared_ptr<TestTensor> TestTensor::matmul(const std::shared_ptr<TestTensor>& b) {
    if (rank < 2  || b->rank != 2)
        throw "NANANA";
    switch (rank) {
        case 2:
            return std::make_shared<TestTensor>(t2.contract(b->t2, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(1, 0)}));
        case 3:
            return std::make_shared<TestTensor>(t3.contract(b->t2, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(2, 0)}));
        case 4:
            return std::make_shared<TestTensor>(t4.contract(b->t2, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(3, 0)}));
        default:
            throw "NANANA";
    }
}

bool TestTensor::needsGradient() {
    return requiresGrad || grad_fn.has_value();
}

void TestTensor::setGradFn(const std::shared_ptr<CNode>& g) {
    grad_fn = std::optional<std::shared_ptr<CNode>>(g);
}