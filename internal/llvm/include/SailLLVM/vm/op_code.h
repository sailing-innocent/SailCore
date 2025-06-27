#pragma once
/**
  * @file op_code.h
  * @brief The OpCode of the Simple Virtual Machine
  * @author sailing-innocent
  * @date 2025-02-04
  */

#include <iostream>
#include "SailLLVM/config.h"

namespace sail::llvm {

enum OpCode {
	OP_SUB,
	OP_ADD,
	OP_MUL,
	OP_PUSH,
	OP_STORE,
	OP_LOAD,
	OP_BNEZ,
	OP_HALT,
};

SAIL_LLVM_API std::ostream& operator<<(std::ostream& os, const OpCode& op);

}// namespace sail::llvm