/*
 * WrappedLiteral.h
 *
 *  Created on: 24 Feb 2012
 *      Author: v1smoll
 */

#ifndef WRAPPEDLITERAL_H_
#define WRAPPEDLITERAL_H_

#include <llvm/Constants.h>

namespace axtor {

class WrappedLiteral
{
public:
	virtual ~WrappedLiteral() {}
	virtual uint getNumOperands() const=0;
	virtual llvm::Constant * getOperand(uint idx) const=0;
};

/*
 * Implements operand based behaviour for ConstantDataSequential-objects
 */
class WrappedDataSequential : public WrappedLiteral
{
	llvm::ConstantDataSequential * constObj;
public:
	WrappedDataSequential(llvm::ConstantDataSequential * _constObj) :
		constObj(_constObj)
	{}

	~WrappedDataSequential() {}

	inline uint getNumOperands() const
	{
		return constObj->getNumElements();
	}

	inline llvm::Constant * getOperand(uint idx) const
	{
		return constObj->getElementAsConstant(idx);
	}
};

/*
 * ConstantVector/Array are already operand based. So, just forward their methods
 */
class WrappedOperandLiteral : public WrappedLiteral
{
	llvm::Constant * constObj;
public:
	WrappedOperandLiteral(llvm::Constant * _constObj) :
		constObj(_constObj)
	{}

	~WrappedOperandLiteral() {}

	inline uint getNumOperands() const
	{
		return constObj->getNumOperands();
	}

	inline llvm::Constant * getOperand(uint idx) const
	{
		return llvm::cast<llvm::Constant>(constObj->getOperand(idx));
	}
};

WrappedLiteral * CreateLiteralWrapper(llvm::Constant * constant)
{
	if (llvm::isa<llvm::ConstantDataSequential>(constant)) {
		return new WrappedDataSequential(llvm::cast<llvm::ConstantDataSequential>(constant));
	} else {
		return new WrappedOperandLiteral(constant);
	}
}

}

#endif /* WRAPPEDLITERAL_H_ */
