#include <iostream>
#include <Eigen/Dense>
#include <spdlog/spdlog.h>
#include "src/ComputationalNode.h"
#include "src/Variable.h"
#include "src/Add.h"
#include "src/Multiply.h"
#include "src/Sub.h"
#include "src/CWiseMultiply.h"
#include "src/ReduceSum.h"
#include "src/Sigmoid.h"
#include <unsupported/Eigen/MatrixFunctions>

using Eigen::MatrixXf;

int main() {
    MatrixXf mTrainX(32, 2);
    MatrixXf mTrainY(32, 1);
    mTrainX << Eigen::Matrix<float, 16, 2>::Random() - Eigen::MatrixXf::Constant(16, 2, 2), Eigen::Matrix<float, 16, 2>::Random() + Eigen::MatrixXf::Constant(16, 2, 1);
    mTrainY << Eigen::MatrixXf::Constant(16, 1, 0), Eigen::MatrixXf::Constant(16, 1, 1);
    auto mW = Eigen::Matrix<float, 2, 1>::Random().cwiseProduct(Eigen::MatrixXf::Constant(2, 1, .1));

    auto W = std::make_shared<Variable>(mW);
    auto tX = std::make_shared<Variable>(mTrainX);
    auto tY = std::make_shared<Variable>(mTrainY);

    auto yp = std::make_shared<Multiply>(tX, W);
    auto ysp = std::make_shared<Sigmoid>(yp);
    auto l1 = std::make_shared<Sub>(ysp, tY);
    auto l2 = std::make_shared<CWiseMultiply>(l1, l1);
    auto l3 = std::make_shared<ReduceSum>(l2);

    for (int i = 0; i < 10000; i++) {
        l3->compute();
        std::cout << l3->mValue << std::endl;
        l3->compute_gradients(MatrixXf::Constant(1, 1, .05));
        // std::cout << W->mGradient << std::endl;
        W->mValue -= W->mGradient;
        W->zero_gradient();
        tX->zero_gradient();
        tY->zero_gradient();
        yp->zero_gradient();
        ysp->zero_gradient();
        l1->zero_gradient();
        l2->zero_gradient();
        l3->zero_gradient();
    }

    std::cout << W->mValue << std::endl;


    //MatrixXf initGrad(1, 1);
    //initGrad << 1;
    //r->compute_gradients(static_cast<Eigen::MatrixBase<float>>(initGrad));

    //std::cout << r->mValue << std::endl;
    //std::cout << *a->mGradient << std::endl;
    //std::cout << x->mGradient << std::endl;

    //spdlog::info("Welcome to spdlog!");
}

