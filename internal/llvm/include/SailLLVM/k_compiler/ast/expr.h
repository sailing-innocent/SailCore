#pragma once
/**
  * @file expr.h
  * @brief The AST for Expression
  * @author sailing-innocent
  * @date 2025-02-04
  */

#include "SailLLVM/config.h"
#include <iterator>
#include <string>
#include <memory>
#include <vector>
#include <iostream>
#include <span>

namespace sail::llvm {

struct NumberExprAST;
struct VariableExprAST;
struct BinaryOpAST;
struct CallOpAST;
struct PrototypeAST;
struct FunctionAST;
struct ForExprAST;

struct SAIL_LLVM_API ExprVisitor {
	virtual void visit(NumberExprAST& node) = 0;
	virtual void visit(VariableExprAST& node) = 0;
	virtual void visit(BinaryOpAST& node) = 0;
	virtual void visit(CallOpAST& node) = 0;
	virtual void visit(ForExprAST& node) = 0;
	virtual ~ExprVisitor() noexcept = default;
};

#define AST_COMMON() \
	void accept(ExprVisitor& visitor) override { visitor.visit(*this); }

class ExprAST {
public:
	virtual ~ExprAST() = default;
	// delete implicit copy and move
	virtual void accept(ExprVisitor& visitor) = 0;
};

class SAIL_LLVM_API NumberExprAST final : public ExprAST {
	int m_val;

public:
	NumberExprAST(int val) : m_val(val) {
		std::cout << "Constructing Number AST " << m_val << "\n";
	}
	int get_val() const { return m_val; }
	AST_COMMON()
};

class VariableExprAST final : public ExprAST {
	std::string m_name;

public:
	VariableExprAST(const std::string& name) : m_name(name) {
		std::cout << "Constructing Variable AST " << m_name << "\n";
	}
	const std::string& get_name() const { return m_name; }
	AST_COMMON()
};

class SAIL_LLVM_API BinaryOpAST final : public ExprAST {
	char m_op;
	std::unique_ptr<ExprAST> m_lhs, m_rhs;

public:
	BinaryOpAST(char op, std::unique_ptr<ExprAST> lhs, std::unique_ptr<ExprAST> rhs)
		: m_op(op), m_lhs(std::move(lhs)), m_rhs(std::move(rhs)) {
		std::cout << "Constructing Binary Op AST " << m_op << "\n";
	}
	std::string get_op() const {
		// ascii char to string
		return std::string(1, m_op);
	}
	ExprAST* get_lhs() const { return m_lhs.get(); }
	ExprAST* get_rhs() const { return m_rhs.get(); }
	AST_COMMON()
};

class SAIL_LLVM_API ForExprAST final : public ExprAST {
	std::string m_var_name;
	std::unique_ptr<ExprAST> m_start, m_end, m_step, m_body;

public:
	ForExprAST(const std::string& var_name,
			   std::unique_ptr<ExprAST> start,
			   std::unique_ptr<ExprAST> end,
			   std::unique_ptr<ExprAST> step,
			   std::unique_ptr<ExprAST> body)
		: m_var_name(var_name),
		  m_start(std::move(start)),
		  m_end(std::move(end)),
		  m_step(std::move(step)),
		  m_body(std::move(body)) {}
	AST_COMMON()

	const std::string& get_var_name() const { return m_var_name; }
	ExprAST* get_start() const { return m_start.get(); }
	ExprAST* get_end() const { return m_end.get(); }
	ExprAST* get_step() const { return m_step.get(); }
	ExprAST* get_body() const { return m_body.get(); }
};

class CallOpAST final : public ExprAST {
	std::string m_callee;
	std::vector<std::unique_ptr<ExprAST>> m_args;

public:
	CallOpAST(const std::string& callee, std::vector<std::unique_ptr<ExprAST>> args)
		: m_callee(callee), m_args(std::move(args)) {
		std::cout << "Constructing CallOpAST\n";
	}
	const std::string& get_callee() const { return m_callee; }
	// get_args: return the iterator for the m_args
	// auto arg : node.get_args())
	std::span<ExprAST*> get_args() const {
		std::vector<ExprAST*> ptrs;
		for (const auto& arg : m_args) {
			ptrs.push_back(arg.get());
		}
		return std::span<ExprAST*>(ptrs);
	}
	AST_COMMON()
};

class PrototypeAST {
	std::string m_name;
	std::vector<std::string> m_args;

public:
	PrototypeAST(const std::string& name, std::vector<std::string> args)
		: m_name(name), m_args(std::move(args)) {
		std::cout << "Constructing Prototype AST: " << m_name << " with " << args.size() << " args\n";
	}
	// delete implicit copy

	const std::string& get_name() const { return m_name; }
	std::vector<std::string> get_args() const {
		return m_args;
	}
};

class FunctionAST {
	std::unique_ptr<PrototypeAST> m_proto;
	std::unique_ptr<ExprAST> m_body;

public:
	FunctionAST(std::unique_ptr<PrototypeAST> proto, std::unique_ptr<ExprAST> body)
		: m_proto(std::move(proto)), m_body(std::move(body)) {
		std::cout << "Constructing Function AST \n";
	}
};

}// namespace sail::llvm