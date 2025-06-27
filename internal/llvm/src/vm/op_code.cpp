/**
  * @file op_code.cpp
  * @brief The OpCode of the Simple Virtual Machine
  * @author sailing-innocent
  * @date 2025-02-04
  */

#include "SailLLVM/vm/op_code.h"

namespace sail::llvm {

std::ostream& operator<<(std::ostream& os, const OpCode& op) {
	switch (op) {
		case OP_SUB: return os << "OP_SUB";
		case OP_MUL: return os << "OP_MUL";
		case OP_PUSH: return os << "OP_PUSH";
		case OP_STORE: return os << "OP_STORE";
		case OP_LOAD: return os << "OP_LOAD";
		case OP_BNEZ: return os << "OP_BNEZ";
		case OP_HALT: return os << "OP_HALT";
	}
	return os;
}

}// namespace sail::llvm