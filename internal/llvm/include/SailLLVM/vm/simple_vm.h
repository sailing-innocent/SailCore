#pragma once
/**
  * @file simple_vm.h
  * @brief The Simple Virtual Machine
  * @author sailing-innocent
  * @date 2025-02-04
  */

#include "SailLLVM/vm/op_code.h"
#include <cstdint>
#include <vector>
#include <stack>
#include <array>

namespace sail::llvm {

class SAIL_LLVM_API SimpleVM {
public:
	SimpleVM(std::vector<int8_t> code) : code_(code) {}
	std::int64_t run();

private:
	std::int64_t pop_value();
	std::vector<int8_t> code_;
	std::stack<int64_t> stack_;
	std::array<int64_t, 256> heap_;
};

}// namespace sail::llvm