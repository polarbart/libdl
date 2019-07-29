
#ifndef LIBDL_CNODEBASE_H
#define LIBDL_CNODEBASE_H


#include <vector>
#include <memory>
#include <stdexcept>


class CNodeBase {
public:
    std::vector<std::shared_ptr<CNodeBase>> children;
    int parentsThatNeedToComputeGradients = 0;
    bool visited = false;
    virtual void computeGradients() = 0;

    static bool noGrad;
protected:
    explicit CNodeBase(std::vector<std::shared_ptr<CNodeBase>>);
};


#endif //LIBDL_CNODEBASE_H
