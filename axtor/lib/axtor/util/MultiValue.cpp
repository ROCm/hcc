/*
 * MultiValue.cpp
 *
 *  Created on: 25.04.2010
 *      Author: gnarf
 */

#include <axtor/util/MultiValue.h>

#include "llvm/Constants.h"

namespace axtor {



	MultiValue::ValueFunctor::ValueFunctor(ValueOperation _op) :
		op(_op)
		{ assert(op && "invalid functions"); }

	void MultiValue::ValueFunctor::operator()(llvm::Value * val) { op(val); }

// ### FUNCTORS ###
void MultiValue::func_dropReferences(llvm::Value * val)
{
	if (llvm::isa<llvm::User>(val))	{
		//std::cerr << "drop ref on : "; val->dump();
		llvm::User * user = llvm::cast<llvm::User>(val);
		user->dropAllReferences();
	}
}

void MultiValue::func_erase(llvm::Value * val)
{

	if (llvm::isa<llvm::GlobalValue>(val))	{
		llvm::GlobalValue * global = llvm::cast<llvm::GlobalValue>(val);
		global->eraseFromParent();

	} else {
		if (llvm::isa<llvm::Instruction>(val)) {
			llvm::Instruction * inst = llvm::cast<llvm::Instruction>(val);
			inst->eraseFromParent();
		}
	}
}

void MultiValue::func_removeConstantUsers(llvm::Value * val)
{
	if (llvm::isa<llvm::GlobalValue>(val)) {
		llvm::GlobalValue * global = llvm::cast<llvm::GlobalValue>(val);

		global->removeDeadConstantUsers();
		for(llvm::Value::use_iterator itUse = global->use_begin(); itUse != global->use_end();)
		{
			llvm::Value * val = *(itUse++);

			if (llvm::isa<llvm::ConstantExpr>(val))
			{
				llvm::ConstantExpr * expr = llvm::cast<llvm::ConstantExpr>(val);

				if (expr->getNumUses() == 0) {
					delete expr;
				}
			}
		}
	}
}

void MultiValue::func_dump(llvm::Value * val)
{
	val->dump();
}

void MultiValue::apply(ValueSet & values, ValueOperation op)
{
	ValueFunctor functor(op);
	std::for_each(values.begin(), values.end(), functor);
}

void MultiValue::erase(ValueSet & values)
{
	apply(values, func_removeConstantUsers);
	apply(values, func_dropReferences);
	apply(values, func_erase);
}

void MultiValue::dump(ValueSet & values)
{
	apply(values, func_dump);
}
}
