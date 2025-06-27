/**
  * @file formatter.cpp
  * @brief The Formatter for AST
  * @author sailing-innocent
  * @date 2025-02-04
  */

#include "SailLLVM/k_compiler/ast/formatter.h"

namespace sail::llvm {

void ExprFormatter::visit(NumberExprAST& node) {
	m_fmt_result += "<NumberExprAST>";
	m_fmt_result += std::to_string(node.get_val());
	m_fmt_result += "</NumberExprAST>";
}

void ExprFormatter::visit(VariableExprAST& node) {
	m_fmt_result += "<VariableExprAST>";
	m_fmt_result += node.get_name();
	m_fmt_result += "</VariableExprAST>";
}

void ExprFormatter::visit(BinaryOpAST& node) {
	m_fmt_result += "<BinaryOpAST>";
	m_fmt_result += node.get_op();
	m_fmt_result += "<LHS>";
	node.get_lhs()->accept(*this);
	m_fmt_result += "</LHS>";
	m_fmt_result += "<RHS>";
	node.get_rhs()->accept(*this);
	m_fmt_result += "</RHS>";
	m_fmt_result += "</BinaryOpAST>";
}

void ExprFormatter::visit(CallOpAST& node) {
	m_fmt_result += "<CallOpAST>";
	m_fmt_result += node.get_callee();
	m_fmt_result += "<Args>";
	for (auto* arg : node.get_args()) {
		arg->accept(*this);
	}
	m_fmt_result += "</Args>";
	m_fmt_result += "</CallOpAST>";
}

void ExprFormatter::visit(ForExprAST& node) {
	m_fmt_result += "<ForExprAST>";
	m_fmt_result += node.get_var_name();
	m_fmt_result += "<Start>";
	node.get_start()->accept(*this);
	m_fmt_result += "</Start>";
	m_fmt_result += "<End>";
	node.get_end()->accept(*this);
	m_fmt_result += "</End>";
	if (node.get_step()) {
		m_fmt_result += "<Step>";
		node.get_step()->accept(*this);
		m_fmt_result += "</Step>";
	}
	m_fmt_result += "<Body>";
	node.get_body()->accept(*this);
	m_fmt_result += "</Body>";
	m_fmt_result += "</ForExprAST>";
}

}// namespace sail::llvm