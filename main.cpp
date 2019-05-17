#include <iostream>
#include <Eigen/Dense>
#include <spdlog/spdlog.h>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

auto f(Tensor<int, 4> x) {
    x.reshape(Eigen::array<int, 4>{2, 2});
}

int main() {
    Tensor<int, 4> x(1, 1, 2, 2);
    auto y = f(x);
    std::cout << y << std::endl;
    Tensor<int, 3> t(4, 2, 2);
    Tensor<int, 2> w(2, 2);
    t.setValues({{{1,2}, {3, 4}}, {{5,6}, {7, 8}}, {{9,10}, {11,12}}, {{13,14}, {15,16}}});
    w.setValues({{1,-1}, {2, -2}});
    (w + t).e
    //t.setConstant(2);
    //w.setConstant(1);
    std::cout << t << std::endl;
    std::cout << w << std::endl;

    std::cout << t.sum() << std::endl;


    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(2, 0) };
    Tensor<int, 3>  p = t.contract(w, product_dims);
    std::cout << p << std::endl;

    /*MatrixXf mTrainX(4, 2);
    MatrixXf mTrainY(4, 1);
    mTrainX <<0, 0, 0, 1, 1, 0, 1, 1;
    mTrainY << 0, 1, 1, 0;

    std::cout << mTrainX.rowwise() + Eigen::Matrix<float, 1, 2>::Ones() << std::endl;

    auto tX = std::make_shared<Variable>(mTrainX);
    auto tY = std::make_shared<Variable>(mTrainY);

    auto W1 = std::make_shared<Variable>(Eigen::Matrix<float, 2, 2>::Random().cwiseProduct(Eigen::MatrixXf::Constant(2, 2, .1)));
    auto b1 = std::make_shared<Variable>(Eigen::Matrix<float, 1, 2>::Zero());
    auto W2 = std::make_shared<Variable>(Eigen::Matrix<float, 2, 1>::Random().cwiseProduct(Eigen::MatrixXf::Constant(2, 1, .1)));
    auto b2 = std::make_shared<Variable>(Eigen::Matrix<float, 1, 1>::Zero());*/

    //MatrixXf initGrad(1, 1);
    //initGrad << 1;
    //r->compute_gradients(static_cast<Eigen::MatrixBase<float>>(initGrad));

    //std::cout << r->mValue << std::endl;
    //std::cout << *a->mGradient << std::endl;
    //std::cout << x->mGradient << std::endl;

    //spdlog::info("Welcome to spdlog!");
}

/*
 * MatrixXf mTrainX(32, 2);
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
 * */