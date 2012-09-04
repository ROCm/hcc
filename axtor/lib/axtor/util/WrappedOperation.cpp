/*
 * WrappedOperation.cpp
 *
 *  Created on: 30 Jan 2012
 *      Author: v1smoll
 */


#include <axtor/util/WrappedOperation.h>

#include <llvm/Instructions.h>
#include <axtor/util/ResourceGuard.h>


namespace axtor {

	WrappedOperation::~WrappedOperation()
	{}

	bool WrappedOperation::isBinaryOp() const
	{
		return llvm::Instruction::isBinaryOp(getOpcode());
	}

	bool WrappedOperation::isCast() const
	{
		return llvm::Instruction::isCast(getOpcode());
	}

	bool WrappedOperation::isCompare() const
	{
		return getOpcode() == llvm::Instruction::ICmp || getOpcode() == llvm::Instruction::FCmp;
	}


	// regular Instruction wrapper
	llvm::User * WrappedInstruction::getValue() const
	{
		return inst;
	}

	uint WrappedInstruction::getOpcode() const
	{
		return inst->getOpcode();
	}

	uint WrappedInstruction::getPredicate() const
	{
		return llvm::cast<llvm::CmpInst>(inst)->getPredicate();
	}

	llvm::Value * WrappedInstruction::getOperand(uint idx) const
	{
		return inst->getOperand(idx);
	}

	uint WrappedInstruction::getNumOperands() const
	{
		return inst->getNumOperands();
	}

	// ConstantExpr wrapper
	llvm::User * WrappedConstExpr::getValue() const
	{
		return expr;
	}

	uint WrappedConstExpr::getOpcode() const
	{
		return expr->getOpcode();
	}

	llvm::Value * WrappedConstExpr::getOperand(uint idx) const
	{
		return expr->getOperand(idx);
	}

	uint WrappedConstExpr::getPredicate() const
	{
		return expr->getPredicate();
	}

	uint WrappedConstExpr::getNumOperands() const
	{
		return expr->getNumOperands();
	}

	template class ResourceGuard<WrappedOperation>;
}
