/**
  * @file k_compiler_with_vm.cpp
  * @brief The K Compiler with VM Backend Sample 
  * @author sailing-innocent
  * @date 2025-02-04
  */

#include <iostream>
#include "SailLLVM/k_compiler/compiler.h"
#include "SailLLVM/k_compiler/ast/formatter.h"
#include "SailLLVM/assembly.h"
#include "SailLLVM/vm/op_code.h"
#include "SailLLVM/vm/simple_vm.h"

int main() {
	using namespace sail::llvm;
	Compiler compiler;
	// std::string code = "a=1\n";
	// std::string code = "a = (a + 1)";
	// std::string code = "a=1\n b=2\n a+b";
	std::string code = "a=1\n for x=1,5,1 (a=(a*x))\n";
	// compiler.compile("def foo(x)\n x + 1\0");
	compiler.compile(code);
	std::cout << "AST:\n";
	std::cout << compiler.exprs.size() << " exprs\n";
	std::cout << "====================\n";
	ExprFormatter formatter;
	for (const auto& expr : compiler.exprs) {
		formatter.visit(expr.get());
	}

	std::cout << formatter.get_fmt_result() << std::endl;

	Assembly assembly;
	for (const auto& expr : compiler.exprs) {
		assembly.visit(expr.get());
	}
	assembly.eof();
	std::cout << "====================\n";
	std::cout << "Assembly:\n";
	std::cout << assembly.m_code.size() << " ops\n";
	for (const auto& op : assembly.m_code) {
		std::cout << std::to_string(op) << " ";
	}
	// // clang-format off
	// std::vector<int8_t> ascode = {
	// 		OP_PUSH, 1,
	// 		OP_STORE, 0,
	// 		OP_PUSH, 2,
	// 		OP_STORE, 1,
	// 		OP_LOAD, 0,
	// 		OP_LOAD, 1,
	// 		OP_ADD,
	// 		OP_HALT
	// };
	// // clang-format off

	// clang-format off
	std::vector<int8_t> ascode = {
			OP_PUSH, 1, OP_STORE, 0,
			OP_PUSH, 1, OP_STORE, 1,
			// LOOP BODY
			OP_LOAD, 0, OP_LOAD, 0, OP_LOAD, 1, OP_MUL,
			OP_STORE, 0, 
			// LOOP BODY END
			OP_LOAD, 1, OP_PUSH, 1, OP_ADD, OP_STORE, 1, // STEP
			OP_PUSH, 5+1+1, OP_LOAD, 1, OP_SUB, OP_BNEZ, -21, // CONDITION
			OP_HALT
	};
	// clang-format on
	SimpleVM vm(assembly.m_code);
	// SimpleVM vm(ascode);
	std::cout << "====================\n";
	std::cout << "Result: " << vm.run() << std::endl;

	return 0;
}