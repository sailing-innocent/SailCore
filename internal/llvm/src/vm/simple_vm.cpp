/**
  * @file simple_vm.cpp
  * @brief The Simple Virtual Machine
  * @author sailing-innocent
  * @date 2025-02-04
  */

#include "SailLLVM/vm/simple_vm.h"

namespace sail::llvm {

std::int64_t SimpleVM::pop_value() {
	auto top = stack_.top();
	stack_.pop();
	return top;
}

std::int64_t SimpleVM::run() {
	for (int pc = 0;; ++pc) {
		std::cout << "-------------------" << std::endl;
		std::cout << "pc: " << pc << " code: " << OpCode(code_[pc]) << std::endl;
		std::cout << "Current stack: ";
		std::stack<int64_t> tmp = stack_;
		while (!tmp.empty()) {
			std::cout << tmp.top() << " ";
			tmp.pop();
		}
		std::cout << std::endl;
		// std::cout << "Current value: " << value_ << std::endl;
		switch (code_[pc]) {
			case OP_ADD: stack_.push(pop_value() + pop_value()); break;
			case OP_SUB: stack_.push(-pop_value() + pop_value()); break;
			case OP_MUL: stack_.push(pop_value() * pop_value()); break;
			case OP_PUSH: stack_.push(code_[++pc]); break;
			case OP_STORE: {
				heap_[code_[++pc]] = pop_value();
				// value_ = pop_value();
				break;
			}
			case OP_LOAD: {
				// stack_.push(value_);
				stack_.push(heap_[code_[++pc]]);
				break;
			}
			case OP_BNEZ: pc += pop_value() ? code_[pc + 1] - 1 : 1; break;
			case OP_HALT: return pop_value();
		}
	}
}

}// namespace sail::llvm