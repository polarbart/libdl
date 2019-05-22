//
// Created by polarbabe on 21.05.19.
//

#include "CNodeBase.h"

CNodeBase::CNodeBase(std::vector<std::shared_ptr<CNodeBase>> p) : children(std::move(p)) {}
