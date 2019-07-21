
#include "CNodeBase.h"

CNodeBase::CNodeBase(std::vector<std::shared_ptr<CNodeBase>> p) : children(std::move(p)) {}
