/**
  * @file assembly.cpp
  * @brief The Assembly for k_compiler_with_vm
  * @author sailing-innocent
  * @date 2025-02-04
  */

#include "SailLLVM/assembly.h"

namespace sail::llvm {

void Assembly::visit(NumberExprAST& node) {
	std::cout << "NumberExprAST: " << node.get_val() << '\n';
	// PUSH to stack
	m_code.push_back(OP_PUSH);
	m_code.push_back(node.get_val());
}

void Assembly::visit(VariableExprAST& node) {
	// add to the variable table
	std::cout << "VariableExprAST: " << node.get_name() << '\n';
	// SSA: Static Single Assignment
	if (m_var_table.find(node.get_name()) == m_var_table.end()) {
		m_var_table[node.get_name()] = m_var_table.size();// the address
	} else {
		// load from memory
		m_code.push_back(OP_LOAD);
		m_code.push_back(m_var_table[node.get_name()]);
	}
}

void Assembly::visit(BinaryOpAST& node) {
	std::cout << "BinaryOpAST: " << node.get_op() << '\n';
	std::cout << ">> LHS: ";
	node.get_lhs()->accept(*this);
	std::cout << ">> RHS: ";
	node.get_rhs()->accept(*this);
	// binary operation
	switch (node.get_op()[0]) {
		case '=': {
			// assignment
			m_code.push_back(OP_STORE);
			// lhs should be a variable expr
			m_code.push_back(m_var_table[static_cast<VariableExprAST*>(node.get_lhs())->get_name()]);
			break;
		}
		case '+': m_code.push_back(OP_ADD); break;
		case '-': m_code.push_back(OP_SUB); break;
		case '*': m_code.push_back(OP_MUL); break;
		// case '/': m_code.push_back(OP_DIV); break;
		default:
			break;
	}
}

void Assembly::visit(CallOpAST& node) {
	// TODO: implement this
}

void Assembly::visit(ForExprAST& node) {
	// early stop when start > end
	auto start_val = static_cast<NumberExprAST*>(node.get_start())->get_val();
	auto end_val = static_cast<NumberExprAST*>(node.get_end())->get_val();
	auto step_val = node.get_step() ? static_cast<NumberExprAST*>(node.get_step())->get_val() : 1;
	if (start_val > end_val) {
		return;
	}

	// var_name, start, end, step, body
	// for i = 1, 10, 1 ( <body> )
	// allocate varname
	auto var_name = node.get_var_name();
	allocate(var_name);
	auto var_addr = m_var_table[var_name];
	// PUSH START
	node.get_start()->accept(*this);
	// STORE i
	m_code.push_back(OP_STORE);
	m_code.push_back(var_addr);
	// LABEL start
	int label_start = m_code.size();
	// =====================================
	// Looping
	// <body>
	node.get_body()->accept(*this);
	// </body>
	// LOAD i
	m_code.push_back(OP_LOAD);
	m_code.push_back(var_addr);
	// step
	if (node.get_step()) {
		node.get_step()->accept(*this);
		m_code.push_back(OP_ADD);
		m_code.push_back(OP_STORE);
		m_code.push_back(var_addr);
	} else {
		m_code.push_back(OP_PUSH);
		m_code.push_back(1);
		m_code.push_back(OP_ADD);
		m_code.push_back(OP_STORE);
		m_code.push_back(var_addr);
	}
	// PUSH END
	// node.get_end()->accept(*this);
	// add start and step
	m_code.push_back(OP_PUSH);
	m_code.push_back(start_val + end_val + step_val);
	// m_code.push_back(OP_ADD);

	// LOAD i
	m_code.push_back(OP_LOAD);
	m_code.push_back(var_addr);
	// COMPARE
	m_code.push_back(OP_SUB);
	// BNEZ end
	m_code.push_back(OP_BNEZ);
	int jump_back = label_start - m_code.size() + 1;
	m_code.push_back(jump_back);
	// =====================================
}

}// namespace sail::llvm