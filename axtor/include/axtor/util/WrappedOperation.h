/*
 * WrappedOperation.h
 *
 *  Created on: 30 Jan 2012
 *      Author: v1smoll
 */

#ifndef WRAPPEDOPERATION_H_
#define WRAPPEDOPERATION_H_

#include <llvm/Value.h>
#include <llvm/Instructions.h>
#include <llvm/Constants.h>
#include <llvm/User.h>
#include <llvm/Type.h>

namespace axtor {

// wrapper class for Users with an Opcode that operated on values
class WrappedOperation
{
public:
	virtual ~WrappedOperation();

	virtual llvm::User * getValue() const=0;
	const llvm::Type * getType() const { return getValue()->getType(); }

	virtual uint getOpcode() const=0;
	virtual llvm::Value * getOperand(uint) const=0;
	virtual uint getNumOperands() const=0;
	virtual uint getPredicate() const=0;

	bool isBinaryOp() const;

	bool isCompare() const;
	bool isCast() const;

	bool isa(uint opcode) const { return getOpcode() == opcode; }
};

// regular Instruction wrapper
class WrappedInstruction : public WrappedOperation
{
	llvm::Instruction * inst;
public:
	WrappedInstruction(llvm::Instruction * _inst) :	inst(_inst) {}
	~WrappedInstruction() {}

	llvm::User * getValue() const;
	uint getOpcode() const;
	uint getPredicate() const;
	llvm::Value * getOperand(uint) const;
	uint getNumOperands() const;
};

// ConstantExpr wrapper
class WrappedConstExpr : public WrappedOperation
{
	llvm::ConstantExpr * expr;
public:
	WrappedConstExpr(llvm::ConstantExpr * _expr) : expr(_expr) {}
	~WrappedConstExpr() {}

	llvm::User * getValue() const;
	uint getOpcode() const;
	uint getPredicate() const;
	llvm::Value * getOperand(uint) const;
	uint getNumOperands() const;
};

}


#endif /* WRAPPEDOPERATION_H_ */
