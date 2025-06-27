#pragma once
/**
  * @file assembly.h
  * @brief The Assembly for k_compiler_with_vm
  * @author sailing-innocent
  * @date 2025-02-04
  */

#include "SailLLVM/vm/simple_vm.h"
#include "SailLLVM/k_compiler/ast/expr.h"
#include <map>

namespace sail::llvm {

class SAIL_LLVM_API Assembly : public ExprVisitor {
	void allocate(const std::string& name) {
		if (m_var_table.find(name) == m_var_table.end()) {
			m_var_table[name] = m_var_table.size();
		}
	}

public:
	Assembly() = default;
	void visit(NumberExprAST& node) override;
	void visit(VariableExprAST& node) override;
	void visit(BinaryOpAST& node) override;
	void visit(CallOpAST& node) override;
	void visit(ForExprAST& node) override;
	void visit(ExprAST* node) {
		node->accept(*this);
	}
	void eof() {
		m_code.push_back(OP_HALT);
	}

	std::vector<int8_t> m_code;
	std::map<std::string, int> m_var_table;
};

}// namespace sail::llvm