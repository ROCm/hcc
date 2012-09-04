/*
 * llvmConstant.h
 *
 *  Created on: 05.03.2010
 *      Author: gnarf
 */

#include <axtor/util/llvmConstant.h>
#include <llvm/Type.h>

#include <llvm/Module.h>

llvm::ConstantInt * axtor::get_uint(uint val)
{
	llvm::APInt apInt(sizeof(int) * 8u, (uint64_t) val, false);
	return llvm::ConstantInt::get(SharedContext::get(), apInt);
}

llvm::ConstantInt * axtor::get_uint(uint val, llvm::LLVMContext & context, llvm::IntegerType * type)
{
	llvm::APInt apInt(type->getScalarSizeInBits(), (uint64_t) val, false);
	return llvm::ConstantInt::get(context, apInt);
}

llvm::ConstantInt * axtor::get_int(uint64_t val, int bits)
{
	llvm::APInt apInt(bits, val, false);
	return llvm::ConstantInt::get(SharedContext::get(), apInt);
}

llvm::ConstantInt * axtor::get_int(int val)
{
	llvm::APInt apInt(sizeof(int) * 8u, (int64_t) val, false);
	return llvm::ConstantInt::get(SharedContext::get(), apInt);
}

llvm::ConstantInt * axtor::get_int(int val, llvm::LLVMContext & context, llvm::IntegerType * type)
{
	llvm::APInt apInt(type->getScalarSizeInBits(), (int64_t) val, false);
	return llvm::ConstantInt::get(context, apInt);
}

llvm::Constant * axtor::get_stringGEP(llvm::Module * module, std::string content)
{
	llvm::LLVMContext & context = module->getContext();

	llvm::Constant * charArray = llvm::ConstantDataArray::getString(context, content, true);
	llvm::Constant * strGlobal = module->getOrInsertGlobal("const", charArray->getType());
	llvm::Constant * zeroConst = llvm::Constant::getNullValue(llvm::Type::getInt32Ty(context));
	llvm::Constant* arr[2];
	arr[0] = zeroConst; arr[1] = zeroConst;
	llvm::Constant * stringGep = llvm::ConstantExpr::getInBoundsGetElementPtr(strGlobal, arr);
	return stringGep;
}

bool axtor::evaluateString(llvm::Value * val, std::string & out)
{
	if (!val || !llvm::isa<llvm::ConstantExpr>(val))
		return false;

	//get constant GEP
	llvm::ConstantExpr * constGEP = llvm::cast<llvm::ConstantExpr>(val);
	if (constGEP->getOpcode() != llvm::Instruction::GetElementPtr)
		return false;

	//get string operand
	llvm::Value * strVal = constGEP->getOperand(0);
	if (! llvm::isa<llvm::GlobalVariable>(strVal))
		return false;

	//TODO verify string type

	//get string constant
	llvm::GlobalVariable * strGlobal = llvm::cast<llvm::GlobalVariable>(strVal);
	llvm::Constant * strConst = strGlobal->getInitializer();
	llvm::ConstantDataArray * strArray = llvm::cast<llvm::ConstantDataArray>(strConst);
	std::string tmp = strArray->getAsString();
	out = tmp.substr(0, tmp.length() - 1);
	return true;
}

bool axtor::evaluateInt(llvm::Value * val, uint64_t & oValue)
{
	bool isInt = llvm::isa<llvm::ConstantInt>(val);

	if (isInt) {
		llvm::ConstantInt * constInt = llvm::cast<llvm::ConstantInt>(val);
		oValue = constInt->getLimitedValue();
		return true;
	}

	return false;
}
