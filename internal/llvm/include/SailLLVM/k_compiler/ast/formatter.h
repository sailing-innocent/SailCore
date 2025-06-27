#pragma once
/**
  * @file formatter.h
  * @brief The Formatter for AST 
  * @author sailing-innocent
  * @date 2025-02-04
  */

#include "SailLLVM/k_compiler/ast/expr.h"

namespace sail::llvm {

class SAIL_LLVM_API ExprFormatter : public ExprVisitor {

public:
	void visit(NumberExprAST& node) override;
	void visit(VariableExprAST& node) override;
	void visit(BinaryOpAST& node) override;
	void visit(CallOpAST& node) override;
	void visit(ForExprAST& node) override;
	std::string m_fmt_result;
	void visit(ExprAST* node) {
		node->accept(*this);
	}
	std::string get_fmt_result() const { return m_fmt_result; }
};

}// namespace sail::llvm