/*
 * OCLWriter.cpp
 *
 *  Created on: 25.04.2010
 *      Author: gnarf
 */

#include <axtor/config.h>
#include <axtor_ocl/OCLWriter.h>

#include <axtor_ocl/OCLEnum.h>
#include <axtor/util/WrappedLiteral.h>

#include "llvm/TypeFinder.h"
#include <vector>

namespace axtor {


 void OCLWriter::put(std::string text)
{
	modInfo.getStream() << text;
}

inline void OCLWriter::putLine(std::string text)
{
	put( text + '\n');
}

inline void OCLWriter::putLineBreak()
{
	modInfo.getStream() <<  std::endl;
}

inline void OCLWriter::put(char c)
{
	put( "" + c );
}

inline void OCLWriter::putLine(char c)
{
	putLine( "" + c );
}

 void OCLWriter::dump()
{}

 void OCLWriter::print(std::ostream & out)
{}

/*
 *  ##### Type System #####
 */

/*
 * returns type symbols for default scalar types
 */
std::string OCLWriter::getScalarType(const llvm::Type * type, bool asVectorElementType)
{
#ifdef DEBUG
	std::cerr << "getScalarType("; type->dump(); std::cerr << ")\n";
#endif

	//### native platform type ###
   std::string nativeStr;
   std::string typeName = "";


   /* NATIVE TYPES */
   if (platform.lookUp(type, typeName, nativeStr))
   {
	   return nativeStr;
   }

	//### C default types ###
  switch (type->getTypeID())
  {
     case llvm::Type::VoidTyID:  return "void";
     case llvm::Type::FloatTyID: return "float";
     case llvm::Type::DoubleTyID: return "double";
     case llvm::Type::IntegerTyID:
     {
        const llvm::IntegerType * intType = llvm::cast<llvm::IntegerType>(type);
        int width = intType->getBitWidth();

        if (width == 1)
           return (asVectorElementType ? "int": "bool");
        else if (width <=8)
           return "char";
        else if (width <= 16)
           return "short";
        else if (width <= 32)
           return "int";
        else if (width <= 64) {
           return "long";
        } else
           Log::fail(type, "(width > 64) over-sized integer type");
     }

  //FIXME
	/*case llvm::Type::OpaqueTyID:
	{
		Log::fail(type, "OpenCL does not implement this opaque type");
	};*/

	case llvm::Type::PointerTyID:
	{
		Log::fail(type, "OpenCL does not support nested pointers");
	}

	case llvm::Type::VectorTyID:
	{
		const llvm::VectorType * vectorType = llvm::cast<llvm::VectorType>(type);
		std::string elementStr = getScalarType(vectorType->getElementType());
		int size = vectorType->getNumElements();

		if (size > 16) {
			Log::fail(type, "OpenCL does not support vector types with more than 16 elements");
		}

		return elementStr + str<int>(size);
	}

     default:
        Log::fail(type, "not a scalar type");
  };
  abort();
}

// FIXME
std::string OCLWriter::getAddressSpaceName(uint space)
{
	switch (space)
	{
	case SPACE_PRIVATE:
		return "__private";
	case SPACE_LOCAL:
		return "__local";
	case SPACE_GLOBAL:
		return "__global";
	case SPACE_CONSTANT:
		return "__constant";
	default:
		Log::warn("encountered unsupported address space (was " + str<uint>(space) + "). Demoting to default adress space.");
		return "";
	};
}

std::string OCLWriter::getLocalVariableKernel(const std::string &variableFullName) {
  return variableFullName.substr(0, variableFullName.find_first_of('.'));
}

std::string OCLWriter::getLocalVariableName(const std::string &variableFullName) {
  return variableFullName.substr(variableFullName.find_first_of('.') + 1, 
                                 variableFullName.length() - 
                                 variableFullName.find_first_of('.'));
}

/*
 * generates a type name for @type
 * if this is a pointer type, operate on its element type instead
 */
std::string OCLWriter::getType(const llvm::Type * type)
{
#ifdef DEBUG
	std::cerr << "getType("; type->dump(); std::cerr << ")\n";
#endif
	if (llvm::isa<llvm::ArrayType>(type)) {
		const llvm::ArrayType * arrType = llvm::cast<llvm::ArrayType>(type);
		return getType(arrType->getElementType()) + "[" + str<int>(arrType->getNumElements()) + "]";

	} else if (llvm::isa<llvm::VectorType>(type)) {
		const llvm::VectorType * vectorType = llvm::cast<llvm::VectorType>(type);
		return getScalarType(vectorType->getElementType(), true) + str<int>(vectorType->getNumElements());

	} else if (llvm::isa<llvm::PointerType>(type)) {
		const llvm::PointerType * ptrType = llvm::cast<llvm::PointerType>(type);
		const llvm::Type * elementType = ptrType->getElementType();
		uint space = ptrType->getAddressSpace();

		if (space == SPACE_NOPTR) {
			return getType(elementType);

		} else {
			std::string addSpaceName = getAddressSpaceName(space);
			if (! addSpaceName.empty()) {
				if (elementType->isFirstClassType())
					return addSpaceName + " " + getType(elementType) + "*";
				else
					return addSpaceName + " (" + getType(elementType) + ")*";

			} else {
				return getType(elementType) + "*";
			}
		}

	} else if (llvm::isa<llvm::StructType>(type)) {

    // Get all of the struct types used in the module.
    llvm::Module *module = modInfo.getModule();
    
    llvm::TypeFinder SrcStructTypes;
    SrcStructTypes.run(*module, true);
    std::vector<llvm::StructType*> structTypes(SrcStructTypes.begin(),
                                                 SrcStructTypes.end());
    
    std::vector<llvm::StructType*>::iterator result = 
      std::find(structTypes.begin(), structTypes.end(), type);
    if (result != structTypes.end())
      return "struct " + (*result)->getName().str();
    else
      Log::fail(type, "anonymous structs not implemented");

		/*if (getTypeSymbol(modInfo.getTypeSymbolTable(), type, name))
		{
			return "struct " + name;
		} else {
			Log::fail(type, "anonymous structs not implemented");
		}*/
	}

	return getScalarType(type);
}

/*
 * build a C-style declaration for @root of type @type
 */
std::string OCLWriter::buildDeclaration(std::string root, const llvm::Type * type)
{
	if (llvm::isa<llvm::ArrayType>(type)) {
		const llvm::ArrayType * arrType = llvm::cast<llvm::ArrayType>(type);
		return buildDeclaration(root + "[" + str<int>(arrType->getNumElements()) + "]", arrType->getElementType());

	} else if (llvm::isa<llvm::VectorType>(type)) {
		const llvm::VectorType * vectorType = llvm::cast<llvm::VectorType>(type);
		return getScalarType(vectorType->getElementType(), true) + str<int>(vectorType->getNumElements()) + " " + root;

	} else if (llvm::isa<llvm::PointerType>(type)) {
		const llvm::PointerType * ptrType = llvm::cast<llvm::PointerType>(type);
		const llvm::Type * elementType = ptrType->getElementType();
		uint space = ptrType->getAddressSpace();

		if ( space == SPACE_NOPTR) { //add address space modifier
			return buildDeclaration(root, elementType);

		} else {
			std::string spaceName = getAddressSpaceName(space);

			if (! spaceName.empty()) {
				std::string ptrPrefix = getAddressSpaceName(space) + "* ";
				if (elementType->isFirstClassType())
					return buildDeclaration(ptrPrefix + root, elementType);
				else
					return buildDeclaration(ptrPrefix + "(" + root + ")", elementType);

			} else {
				return buildDeclaration("* " + root, elementType);
			}
		}

	} else if (llvm::isa<llvm::StructType>(type)) {
		std::string name;
    // Get all of the struct types used in the module.
    llvm::Module *module = modInfo.getModule();
    
    llvm::TypeFinder SrcStructTypes;
    SrcStructTypes.run(*module, true);
    std::vector<llvm::StructType*> structTypes(SrcStructTypes.begin(),
                                                 SrcStructTypes.end());
    
    std::vector<llvm::StructType*>::iterator result =
      std::find(structTypes.begin(), structTypes.end(), type);

    if (result != structTypes.end())
      return "struct " + (*result)->getName().str() + " " + root;
    else
      Log::fail(type, "anonymous structs not implemented");
	}

	std::string scalarStr = getScalarType(type);
	return scalarStr + " " + root;
}

/*
 * ##### DECLARATIONS / OPERATORS & INSTRUCTIONS ######
 */

	/*
* writes a generic function header and declares the arguments as mapped by @locals
*/
std::string OCLWriter::getFunctionHeader(llvm::Function * func, IdentifierScope * locals)
{
	std::stringstream builder;
	const llvm::FunctionType * type = getFunctionType(func);
	const llvm::Type * returnType = type->getReturnType();

	//attributes
	if (modInfo.isKernelFunction(func)) {
		builder << "__kernel ";
	} else if (func->hasFnAttr(llvm::Attribute::InlineHint | llvm::Attribute::AlwaysInline)) {
		builder << "inline ";
	}


	//return type
	std::string typeStr = getType(returnType);
	builder << typeStr;

	//function name
	builder << ' ' << func->getName().str();

	//arguments
	ArgList & argList = func->getArgumentList();
	ArgList::iterator arg;


	//catch empty lists
	if (argList.empty())
	{
		builder << "()";
		return builder.str();
	}

	//create generic argument list
	uint i;
	for(i = 0, arg = argList.begin();
		i < type->getNumParams() && arg != argList.end();
		++i, ++arg)
	{
		const llvm::Type * argType = type->getParamType(i);

		// dereference byVal pointers
		if (arg->hasByValAttr()) {
			argType = llvm::cast<llvm::PointerType>(argType)->getElementType();
		}

		//std::string modifierStr = inferModifiers(arg);
		std::string typeStr = getType(argType);
		std::string argName = typeStr;

		if (locals) {
			const VariableDesc * desc = locals->lookUp(arg);
			argName += ' ' + desc->name;
		}

		 if (arg == argList.begin()) {
			 builder << '(';
		 } else {
			 builder << ", ";
		 }

		 builder << argName;
	}

	builder << ')';
	return builder.str();
}

std::string OCLWriter::getFunctionHeader(llvm::Function * func)
{
	return getFunctionHeader(func, NULL);
}

 void OCLWriter::writeLineBreak()
{
	putLineBreak();
}

 void OCLWriter::writeVariableDeclaration(const VariableDesc & desc)
{
	const llvm::Type * type = desc.type;
	if (desc.isAlloca) {
		type = desc.type->getContainedType(0);
	}

	putLine( buildDeclaration(desc.name, type )  + ";");
}

 void OCLWriter::writeFunctionDeclaration(llvm::Function * func, IdentifierScope * locals)
{
	putLine( getFunctionHeader(func, locals) + ';');
}

/*
 * default C Style operators
 */
std::string OCLWriter::getOperatorToken(const WrappedOperation & op, bool & isSigned)
{
  isSigned = true;
  switch (op.getOpcode())
  {
     //Arithmetic
  	 case llvm::Instruction::FAdd:
     case llvm::Instruction::Add: return "+";
     case llvm::Instruction::FMul:
     case llvm::Instruction::Mul: return "*";
     case llvm::Instruction::FSub:
     case llvm::Instruction::Sub: return "-";
     case llvm::Instruction::UDiv:
      isSigned = false;
     case llvm::Instruction::SDiv:
     case llvm::Instruction::FDiv: return "/";

     case llvm::Instruction::URem:
    	 isSigned = false;
     case llvm::Instruction::SRem:
     case llvm::Instruction::FRem: return "%";

 //binary integer ops
     case llvm::Instruction::Shl: return "<<";
     case llvm::Instruction::LShr: return ">>";
     case llvm::Instruction::AShr: return ">>";

     //the distinction between logic and bitwise operators is based on the instruction type (if its a "bool" aka "i1" return the logic operator)
     case llvm::Instruction::And:
     case llvm::Instruction::Or:
     case llvm::Instruction::Xor:
     {
    	 if (op.getType()->isIntegerTy(1)) {
    		 switch (op.getOpcode()) {
				 case llvm::Instruction::And: return "&&";
				  case llvm::Instruction::Or: return "||";
				  case llvm::Instruction::Xor: return "!=";
			 };

    	 } else {
    		 switch (op.getOpcode()) {
				 case llvm::Instruction::And: return "&";
				  case llvm::Instruction::Or: return "|";
				  case llvm::Instruction::Xor: return "^";
			 };
    	 }
     }

     //predicated CmpInsts
     case llvm::Instruction::FCmp:
     {
        switch (op.getPredicate())
        {
           case llvm::CmpInst::FCMP_FALSE:
             assert(false && "why did nobody optimize that out?");

           case llvm::CmpInst::FCMP_OGT:
           case llvm::CmpInst::FCMP_UGT:
              return ">";

           case llvm::CmpInst::FCMP_OGE:
           case llvm::CmpInst::FCMP_UGE:
              return ">=";

           case llvm::CmpInst::FCMP_OLT:
           case llvm::CmpInst::FCMP_ULT:
              return "<";

           case llvm::CmpInst::FCMP_OLE:
           case llvm::CmpInst::FCMP_ULE:
              return "<=";

           case llvm::CmpInst::FCMP_OEQ:
              return "=="; //overloaded operator in C

           default:
              assert(false && "unmapped cmp predicate");
        };
     }

     case llvm::Instruction::ICmp:
     {
        switch (op.getPredicate())
        {
           case llvm::CmpInst::ICMP_UGT:
        	   isSigned = false;
           case llvm::CmpInst::ICMP_SGT:
              return ">";

           case llvm::CmpInst::ICMP_UGE:
        	   isSigned = false;
           case llvm::CmpInst::ICMP_SGE:
              return ">=";

           case llvm::CmpInst::ICMP_ULT:
        	   isSigned = false;
           case llvm::CmpInst::ICMP_SLT:
              return "<";

           case llvm::CmpInst::ICMP_ULE:
        	   isSigned = false;
           case llvm::CmpInst::ICMP_SLE:
              return "<=";

           case llvm::CmpInst::ICMP_EQ:
              return "=="; //overloaded operator in C

           case llvm::CmpInst::ICMP_NE:
              return "!=";

           default:
              assert(false && "unmapped cmp predicate");
        };
     }

     default:
    	 Log::fail(op.getValue(), str<uint>(op.getOpcode()) + " unsupported operator type");
  }
  Log::fail("failure : internal error in getOperator()");
  return "UNREACHABLE";
}

std::string OCLWriter::getInstruction(llvm::Instruction * inst, std::vector<std::string> operands)
{
	return getOperation(WrappedInstruction(inst), operands);
}

std::string OCLWriter::getOperation(const WrappedOperation & op, StringVector operands)
{
  std::string tmp;

  //# binary infix operator
  if (op.isBinaryOp() || op.isCompare()) {
	  bool signedOps = true;
     std::string token = getOperatorToken(op, signedOps);

     if (signedOps) {
    	 return "(" + operands[0] + token + operands[1] + ")";

     } else {
    	 const llvm::Type * operandType = op.getOperand(0)->getType();
    	 std::string opTypeStr = getType(operandType);
    	 std::string convUnsignedStr = "as_u" + opTypeStr;

    	 return  "(" +
    			 	convUnsignedStr + "(" + operands[0] + ") " + token + " " +
    			    convUnsignedStr + "(" + operands[1] + "))";

     }

  //# bitcast
  } else if (op.isa(llvm::Instruction::BitCast)) {
	 const llvm::Type * srcType = op.getOperand(0)->getType();
	 const llvm::Type * destType = op.getType();

     assert(operands.size() == 1 && "cast a single value . . non?");

     //fake assignment instruction (cast to same type)
     if (srcType == destType) {
    	 return operands[0];

     } else {
    	 std::string typeStr = getType(destType);

    	 if (srcType->isPointerTy() && destType->isPointerTy()) {
    		 return "(" + typeStr + ")(" + operands[0] + ")";
    	 } else {
    		 return "as_" + typeStr + "(" + operands[0] + ")";
    	 }
     }

  //# reinterpreting cast
  } else if (op.isCast()) {
	  const llvm::Type * sourceType = op.getOperand(0)->getType();
	  const llvm::Type * targetType = op.getType();

	  if (op.isa(llvm::Instruction::UIToFP)) { //integers are declared signed
		  std::string intCast = "u" + getType(sourceType);
		  return "convert_" + getType(targetType) + "(as_" + intCast + "(" + operands[0] + "))";

	  // truncation: mask out bits and cast to smaller type
	  } else if (op.isa(llvm::Instruction::Trunc)) {
		  const llvm::Type * destIntType = llvm::cast<llvm::IntegerType>(targetType);
		  uint destWidth = destIntType->getPrimitiveSizeInBits();

		  uint64_t maskInt = generateTruncMask(destWidth);

		  std::string fittedStr = operands[0] + " & 0x" + convertToHex(maskInt, std::max<int>(1, destWidth / 4));

		  //convert_bool is not supported
		  return (destWidth == 1 ? "(bool)" : "convert_" + getType(targetType)) + "(" + fittedStr + ")";

	  } else if (! targetType->isIntegerTy(1)){ //use ints for bools
		  bool isUnsigned = op.isa(llvm::Instruction::ZExt);
		  std::string targetTypeStr = getType(targetType);
		  std::string srcTypeStr = getType(sourceType);

		  // special cast bool to int case
		  if (sourceType->isIntegerTy(1)) {
			  std::string targetCastStr;
			  std::string suffixStr;
			  if (isUnsigned) {
				  targetCastStr =  "as_" + targetTypeStr + "((u" + targetTypeStr + ")("; suffixStr += "))";
			  } else {
				  targetCastStr =  "(" + targetTypeStr + ")(" ; suffixStr += ")";
			  }
			  return targetCastStr + operands[0] + suffixStr;
		  }

		  // we need to operate on unsigned data types to get a zero extension
		  if (isUnsigned) {
			  std::string srcCastStr = "(as_u" + srcTypeStr;
			  std::string targetCastStr =  "as_" + targetTypeStr + "(convert_u" + targetTypeStr;
			  return targetCastStr + srcCastStr + "(" + operands[0] + ")))";

		 // bool conversions and sign/float extension will do without casts (hopefully)
		  } else {
		  	  return "convert_" + getType(targetType) + "(" + operands[0] + ")";
		  }
	  }

  //# select
  } else if (op.isa(llvm::Instruction::Select)) {
	  return operands[0] + "? " + operands[1] + ": " + operands[2];

  //# function call
  } else if (op.isa(llvm::Instruction::Call)) {
	  	assert(!llvm::isa<llvm::ConstantExpr>(op.getValue()) && "do not implemented ConstantExpr-calls");

		llvm::CallInst * caller = llvm::cast<llvm::CallInst>(op.getValue());
		llvm::Function * callee = caller->getCalledFunction();

		StringVector::const_iterator beginParams = operands.begin();
		StringVector::const_iterator endParams = operands.end();

	  if (platform.implements(callee))
	  {
		  return platform.build(callee->getName().str(), beginParams, endParams);

	  } else {
		  tmp = callee->getName().str(); //add function name
	  }

	 tmp += '(';
		 for(StringVector::const_iterator itOp = beginParams; itOp != endParams; ++itOp)
		 {
			if (itOp != beginParams)
			   tmp +=", " + *itOp;
			else
			   tmp += *itOp;
		 }
	 tmp += ')';
	 return tmp;

  //# ERROR handling (foremost generic opcode based scheme)
  }

  Log::fail(std::string(llvm::Instruction::getOpcodeName(op.getOpcode())) + "unimplemented instruction type");
  abort();
}


typedef std::vector<llvm::Value*> ValueVector;


std::string OCLWriter::buildArraySubscript(std::string root, AddressIterator *& address, IdentifierScope & locals)
{
	uint64_t index;

	std::stringstream out;
	bool isFirst = true;

	do
	{
		llvm::Value * indexVal = address->getValue();

		if (!isFirst) {
			out << " + ";
		}

		//dynamic subscript
		if (evaluateInt(indexVal, index)) {
			out << str<uint64_t>(index);

		//static subscript
		} else {
			const VariableDesc * desc = locals.lookUp(indexVal);
			assert(desc && "undefined index value");
			out << desc->name;
		}

		isFirst = false;
		address = address->getNext();
	} while (address && address->isCumulative());

	return root + "[" + out.str() + "]";
}

std::string OCLWriter::getReferer(llvm::Value * value, IdentifierScope & locals)
{
	const VariableDesc * desc = locals.lookUp(value);

	if (desc) {
		return desc->name;
	}

	//not a instruction -> obtain a referer by other means
	return getNonInstruction(value, locals);
}

/*
* return a dereferencing string for the next type node of the object using address
*/
std::string OCLWriter::dereferenceContainer(std::string root, const llvm::Type * type, AddressIterator *& address, IdentifierScope & locals, const llvm::Type *& oElementType, uint addressSpace)
{
	if (llvm::isa<llvm::StructType>(type)) {
		uint64_t index;
		llvm::Value * indexVal = address->getValue();

		if (! evaluateInt(indexVal, index)) {
			Log::fail(type, "can not dynamically access struct members");
		}

		address = address->getNext();
		oElementType = type->getContainedType(index);


		return "(" + root + ").x" + str<int>(index);

	} else if (llvm::isa<llvm::ArrayType>(type)) {
		oElementType = type->getContainedType(0);
		return buildArraySubscript(root, address, locals);

	} else if (llvm::isa<llvm::PointerType>(type)) {
		const llvm::PointerType * ptrType = llvm::cast<llvm::PointerType>(type);
		llvm::Value * indexVal = address->getValue();
		oElementType = type->getContainedType(0);

		if (ptrType->getAddressSpace() == SPACE_NOPTR) {
			address = address->getNext();
			uint64_t index;

			if (!evaluateInt(indexVal,index) || (index != 0)) {
				Log::fail(ptrType, "can not index to into a NOPTR address space value (may only dereference it directly)");
			}

			return root;

		} else {
			// address = address->getNext();
			return buildArraySubscript(root, address, locals);

		}
	// cast to pointer and dereference from there (slightly hacky)
	} else if (llvm::isa<llvm::VectorType>(type)) {
		const llvm::VectorType * vecType = llvm::cast<llvm::VectorType>(type);
		llvm::Type * elemType = type->getContainedType(0);
		uint width = vecType->getNumElements();
		llvm::ArrayType * arrType = llvm::ArrayType::get(elemType, width);
		llvm::PointerType * ptrToElemType = llvm::PointerType::get(elemType, addressSpace);

		std::string castRootStr = "((" + getType(ptrToElemType) +")&(" + root + "))";

		return dereferenceContainer(castRootStr, arrType, address, locals, oElementType, addressSpace);

	}

	Log::fail(type, "can not dereference this type");
}

std::string OCLWriter::getVolatilePointerTo(llvm::Value * val, IdentifierScope & locals, const std::string * rootName)
{
	std::string ptrStr = getPointerTo(val, locals, rootName);
	const llvm::Type * type = val->getType();
	std::string castStr = "volatile " + getType(type);
	return "(" + castStr + ")(" + ptrStr + ")";
}

std::string OCLWriter::getPointerTo(llvm::Value * val, IdentifierScope & locals, const std::string * rootName)
{
	bool isDereffed;
	std::string core = unwindPointer(val, locals, isDereffed, rootName);
	std::string ptrStr;
	if (isDereffed) {
		ptrStr =  "&(" + core + ")";
	} else {
		ptrStr = core;
	}

	return ptrStr;
}

std::string OCLWriter::getReferenceTo(llvm::Value * val, IdentifierScope & locals, const std::string * rootName)
{
	bool isDereffed;
	std::string core = unwindPointer(val, locals, isDereffed, rootName);
	if (isDereffed) {
		return core;
	} else {
		return "*(" + core + ")";
	}
}

/*
 * return a name representing a dereferenced pointer
 *if noImplicitDeref is false, the exact address of the value is returned
 *
 *@param rootName  if set, the rootName is used as base-string for dereferencing, otw the detected base value is look-up in @locals
 *@param oDereferenced returns if the pointer is already dereferenced
 */

std::string OCLWriter::unwindPointer(llvm::Value * val, IdentifierScope & locals, bool & oDereferenced, const std::string * rootName)
{
	const llvm::PointerType * ptrType = llvm::cast<llvm::PointerType>(val->getType());

	uint addressSpace = ptrType->getAddressSpace();
	AddressIterator::AddressResult result = AddressIterator::createAddress(val, platform.getDerefFuncs());
	ResourceGuard<AddressIterator> __guardAddress(result.iterator);

	AddressIterator * address = result.iterator;
	llvm::Value * rootValue = result.rootValue;

	const llvm::Type * rootType = rootValue->getType();

#ifdef DEBUG
	std::cerr << "dereferencing value:\n";
	val->dump();
	std::cerr << "root value:\n";
	rootValue->dump();
	std::cerr << "root type:\n";
	rootType->dump();
	std::cerr << "address:\n";
	if (address)address->dump();
	else std::cerr << "NONE\n";
#endif

	//allocas are dereffed by their name string
	bool hasImplicitPtrDeref = llvm::isa<llvm::AllocaInst>(rootValue);

	//byval function arguments don't need be dereferenced
	if (llvm::isa<llvm::Argument>(rootValue))
	{
		llvm::Argument * arg = llvm::cast<llvm::Argument>(rootValue);
		hasImplicitPtrDeref |= arg->hasByValAttr();
	}

	//local variables are initialised in the program, so assume implicit deref
	if (llvm::isa<llvm::GlobalVariable>(rootValue)) {
		llvm::GlobalVariable * gv = llvm::cast<llvm::GlobalVariable>(rootValue);
		hasImplicitPtrDeref |= gv->isConstant() || gv->getType()->getAddressSpace() == SPACE_LOCAL || gv->getType()->getAddressSpace() == SPACE_CONSTANT;
	}

	// this variable is implicitly dereferenced by its designator
	if (hasImplicitPtrDeref) {
		if (address) {
			uint64_t test;
			assert(evaluateInt(address->getValue(), test) && test == 0 && "skipped non-trivial dereferencing value");
			address = address->getNext();
		}
		rootType = rootType->getContainedType(0);
	}

	//dereff the initial pointer (if it points to a more complex structure)
	std::string tmp;
	if (rootName) {
		tmp = *rootName;

	} else if (llvm::isa<llvm::ConstantExpr>(rootValue)) {
		tmp = getConstant(llvm::cast<llvm::Constant>(rootValue), locals);

	} else {
		const VariableDesc * desc = locals.lookUp(rootValue);
		assert(desc && "root value was not mapped");
		tmp = desc->name;
	}

	  // this is a pointer
	  if (!address) {
		  oDereferenced = hasImplicitPtrDeref;
		  return tmp;
	  }


	while (address)
	{
#ifdef DEBUG
		std::cerr << "deref : " << tmp << "\n";
		assert(rootType && "was not set");
		rootType->dump();
#endif
		const llvm::Type * elementType = 0;
		tmp = dereferenceContainer(tmp, rootType, address, locals, elementType, addressSpace);
		//assert(elementType && "derefContainer did not set element type");
		rootType = elementType;
#ifdef DEBUG
		std::cerr << "dereferenced to " << tmp << "\n";
#endif

	}

	oDereferenced = true;

	return tmp;
}

	std::string OCLWriter::getAllNullLiteral(const llvm::Type * type)
	{
		switch(type->getTypeID())
		{
			case llvm::Type::VectorTyID:
			{
				const llvm::VectorType * arrType = llvm::cast<llvm::VectorType>(type);
				//uint size = arrType->getNumElements();
				std::string elementStr = getAllNullLiteral(arrType->getElementType());
				return "(" + getType(type) + ")(" + elementStr + ")";
			}



			//case llvm::Type::StructTyID:
			//case llvm::Type::ArrayTyID:

/*					const llvm::ArrayType * arrType = llvm::cast<llvm::ArrayType>(type);
				uint size = arrType->getNumElements();
				std::string elementStr = getAllNullLiteral(arrType->getElementType());

				std:string accu = "{";
				for(int i = 0; i < size; ++i)
				{
					if (i > 0) accu += ", ";
					accu += elementStr;
				}
				accu += "}";
				return accu;*/

			case llvm::Type::DoubleTyID:
				return "0.0f";

			case llvm::Type::FloatTyID:
				return "0.0f";

			case llvm::Type::IntegerTyID:
				return "0";

			default:
				Log::fail(type, "OpenCL does not natively support null literals of this kind");
		}
	}

	/*
    * return the string representation of a constant
    */
   std::string OCLWriter::getLiteral(llvm::Constant * val)
   {
      //## Constant integer (and Bool)
      if (llvm::isa<llvm::ConstantInt>(val))
      {
    	  const llvm::IntegerType * intType = llvm::cast<llvm::IntegerType>(val->getType());
    	  if (intType->getBitWidth() == 1) {
    		  if (val->isNullValue()) {
    			  return "false";
    		  } else {
    			  return "true";
    		  }
    	  } else if (intType->getBitWidth() <= 64) {
    		  llvm::ConstantInt * constInt = llvm::cast<llvm::ConstantInt>(val);
			  uint64_t data = constInt->getLimitedValue();
			  std::string hexStr = convertToHex(data, intType->getBitWidth() / 4);
			  std::string typeStr = getScalarType(intType);

			  return "(" + typeStr + ")(0x" + hexStr + ")";
    	  } else {
                 Log::fail(val, "value exceeds size limit, expected bit size <= 64, was " + str<int>(intType->getBitWidth()));
    	  }

      //## Constant Float
      } else if (llvm::isa<llvm::ConstantFP>(val)) {
         llvm::ConstantFP * constFP = llvm::cast<llvm::ConstantFP>(val);
         llvm::APFloat apFloat = constFP->getValueAPF();
         if (&apFloat.getSemantics() == &llvm::APFloat::IEEEsingle) {
        	 float num = apFloat.convertToFloat();
        	 return "(float)(" + str<float>(num) + ")";
         } else if (&apFloat.getSemantics() == &llvm::APFloat::IEEEdouble) {
        	 double num = apFloat.convertToDouble();
        	 return "(double)(" + str<double>(num) + ")";
         } else {
        	 Log::fail(val, "Unsupported constant float");
         }


      //## Function
      } else if (llvm::isa<llvm::Function>(val)) {
         return val->getName();

      //## Constant Array
      } else if (llvm::isa<llvm::ConstantArray>(val) ||
    		  llvm::isa<llvm::ConstantDataArray>(val)) {

    	 ResourceGuard<WrappedLiteral> arr(CreateLiteralWrapper(val));

         std::string buffer = "{";
         const llvm::ArrayType * arrType = llvm::cast<llvm::ArrayType>(val->getType());
         for(uint i = 0; i < arrType->getNumElements(); ++i) {
            llvm::Constant * elem = arr->getOperand(i);
            if (i > 0)
               buffer += "," + getLiteral(elem);
            else
               buffer += getLiteral(elem);
         }
         buffer += "}";
         return buffer;

      } else if (llvm::isa<llvm::ConstantVector>(val) ||
    		  llvm::isa<llvm::ConstantDataVector>(val)) {

    	  ResourceGuard<WrappedLiteral> vector(CreateLiteralWrapper(val));

    	  EXPENSIVE_TEST if (! vector.get())
    	  {
    		  Log::fail(val, "unrecognized constant type");
    	  }

    	  const llvm::VectorType * vectorType = llvm::cast<llvm::VectorType>(val->getType());

    	  std::string buffer = "";
    	  for(uint i = 0; i < vectorType->getNumElements(); ++i)
    	  {
    		  llvm::Value * opVal = vector->getOperand(i);
    		  std::string opStr = getLiteral(llvm::cast<llvm::Constant>(opVal));

    		  if (i > 0) {
    			  buffer += ", ";
    		  }
    		  buffer += opStr;
    	  }

    	  return "(" + getType(vectorType) + ")(" + buffer + ")";

      //default undefined values to zero initializers
      } else if (llvm::isa<llvm::UndefValue>(val) || val->isNullValue()) {
    	  const llvm::Type * type = val->getType();
    	  return getAllNullLiteral(type);
      }

      //## unsupported literal
      Log::fail(val, "unsupported literal");
      assert(false);
   }

   /*
    * tries to create a literal string it @val does not have a variable
    */
   std::string OCLWriter::getValueToken(llvm::Value * val, IdentifierScope & locals)
   {
	   const VariableDesc * desc = locals.lookUp(val);
	   if (desc) {
		   return desc->name;
	   } else {
		   assert(llvm::isa<llvm::Constant>(val) && "undeclared value is not a constant");
		   return getLiteral(llvm::cast<llvm::Constant>(val));
	   }
   }

   /*
   * returns the string representation of a non-instruction value
   */
   std::string OCLWriter::getNonInstruction(llvm::Value * value, IdentifierScope & locals)
   {
	   assert(!llvm::isa<llvm::Instruction>(value) && "should only be called for non-instruction values");

	   const VariableDesc * desc = NULL;

	   if (llvm::isa<llvm::GlobalValue>(value) && (desc = locals.getParent()->lookUp(value))) {
		   return desc->name;

	   } else if (llvm::isa<llvm::PHINode>(value)) { // PHI-Node
		   const VariableDesc * phiDesc = locals.lookUp(value);
		   assert(phiDesc && "unmapped PHI-Node");
		   return phiDesc->name;

	   } else if (llvm::isa<llvm::Constant>(value)) {
		   return getConstant(llvm::cast<llvm::Constant>(value), locals);
	  }

	 Log::fail(value, "failure : could not translate nonInstruction");
	 return "";
   }


   std::string OCLWriter::getConstant(llvm::Constant * constant, IdentifierScope & locals)
   {
	   if (llvm::isa<llvm::ConstantExpr>(constant)) { //arbitrary constant
		   llvm::ConstantExpr * expr = llvm::cast<llvm::ConstantExpr>(constant);
		   if (expr->getOpcode() == llvm::Instruction::GetElementPtr) {
			   return getPointerTo(constant, locals);

		   } else {
			   StringVector operands(expr->getNumOperands());
			   for(uint i = 0; i < expr->getNumOperands(); ++i)
			   {
				   operands[i] = getNonInstruction(expr->getOperand(i), locals);
			   }
			   return getOperation(WrappedConstExpr(expr), operands);

		   }
	   } else {
		   return getLiteral(constant);
	   }
   }

  /*
   * returns the string representation of a ShuffleInstruction
   */
std::string OCLWriter::getShuffleInstruction(llvm::ShuffleVectorInst * shuffle, IdentifierScope & locals)
{
	llvm::Value * firstVector = shuffle->getOperand(0);
	const llvm::VectorType * firstType = llvm::cast<const llvm::VectorType>(firstVector->getType());
	llvm::Value * secondVector = shuffle->getOperand(1);
	//const llvm::VectorType * secondType = llvm::cast<const llvm::VectorType>(secondVector->getType());
	llvm::Value * indexVector = shuffle->getOperand(2);
	const llvm::VectorType * indexType =  llvm::cast<const llvm::VectorType>(indexVector->getType());

	llvm::Type * elementType = firstType->getElementType();

	int secondBase = firstType->getNumElements();
	int numIndices = indexType->getNumElements();

	std::string firstStr = getValueToken(firstVector, locals);

	const VariableDesc * secondDesc = locals.lookUp(secondVector);
	bool hasSecond = ! llvm::isa<llvm::UndefValue>(secondVector);

	std::string secondStr;
	if (hasSecond) {
		secondStr = secondDesc ? secondDesc->name : getLiteral(llvm::cast<llvm::Constant>(secondVector));
	}

#ifdef DEBUG
	std::cerr << "SHUFFLE:\n"
			<< "first=" << firstStr << "\n"
			<< "second=" << secondStr << "\n"
			<< "\n";
#endif

	//get the target types name
	std::string typeStr = getType(shuffle->getType());

	//build a string extracting values from one of the two vectors
	std::string accu = "";
	std::string elements = firstStr + ".s";

	int firstMask = shuffle->getMaskValue(0);
	bool wasLiteral = false;
	bool useFirst = firstMask < secondBase;

	if (firstMask >= 0) {
		if (useFirst) {
			elements = firstStr + ".s";
		} else {
			elements = secondStr + ".s";
		}
		wasLiteral = false;
	} else {
		elements = getLiteral(llvm::Constant::getNullValue(elementType));
		wasLiteral = true;
	}

	for(int i = 0; i < numIndices; ++i)
	{
		int mask = shuffle->getMaskValue(i);

		//set up the element source (last was literal, current is literal unlike the last OR change of source vector)
		if (wasLiteral || (mask < 0) || (useFirst != (mask < secondBase))) {
			accu += elements;
			wasLiteral = false;

			if (mask >= 0) {
				useFirst = mask < secondBase;
				if (useFirst) {
					elements = ", " + firstStr + ".s";
				} else {
					assert(hasSecond && "trying to access elements from undef vector");
					elements = ", " + secondStr + ".s";
				}
			}
		}

		//pick elements
		if (mask < 0) {
			wasLiteral = true;
			elements = ", " + getLiteral(llvm::Constant::getNullValue(elementType));
		} else {
			wasLiteral = false;
			if (useFirst) {
				elements += hexstr(mask);
			} else {
				elements += hexstr(mask - secondBase);
			}
		}
	}

	//add last element
	accu += elements;

	return "(" + typeStr + ")(" + accu + ")";
}

/*
 * returns the string representation of an ExtractElementInstruction
 */
std::string OCLWriter::getExtractElementInstruction(llvm::ExtractElementInst * extract, IdentifierScope & locals)
{
	llvm::Value * indexVal= extract->getIndexOperand();
	llvm::Value * vectorVal = extract->getVectorOperand();
	std::string vectorStr = getValueToken(vectorVal, locals);

	uint64_t index;

	if (evaluateInt(indexVal, index)) {
		return vectorStr + ".s" + hexstr(index);
	}

	Log::fail(extract, "can not randomly extract values from a vector");
	abort();
}

void OCLWriter::writeInsertValueInstruction(llvm::InsertValueInst * insert, IdentifierScope & locals)
{
	llvm::Value * derefValue = insert->getOperand(0);
	llvm::Value * insertValue = insert->getOperand(1);


	const VariableDesc * insertDesc = locals.lookUp(insert);
	assert(insertDesc && "instruction not bound");
	const VariableDesc * targetDesc = locals.lookUp(derefValue);

	//assign initial value
	if (targetDesc) {
		writeAssign(*insertDesc, *targetDesc);\

	} else if (! llvm::isa<llvm::UndefValue>(derefValue)) {
		assert(llvm::isa<llvm::Constant>(derefValue));\
		llvm::Constant * derefConst = llvm::cast<llvm::Constant>(derefValue);
		writeAssignRaw(insertDesc->name, getLiteral(derefConst));
	}

	std::string derefStr = getReferenceTo(insert, locals, &insertDesc->name); //the original value was written to the InsertValueInstructions variable

	//insert value
	std::string valueStr;
	const VariableDesc * valueDesc = locals.lookUp(insertValue);

	if (valueDesc) {
		valueStr = valueDesc->name;
	} else {
		assert(llvm::isa<llvm::Constant>(insertValue));
		llvm::Constant * insertConst = llvm::cast<llvm::Constant>(insertValue);
		valueStr = getLiteral(insertConst);
	}

	writeAssignRaw(derefStr, valueStr);
}

/*
 * returns the string representation of an InsertElementInstruction
 * if the vector value is defined this creates two instructions
 */
void OCLWriter::writeInsertElementInstruction(llvm::InsertElementInst * insert, IdentifierScope & locals)
{
	llvm::Value * vec = insert->getOperand(0);
	llvm::Value * value = insert->getOperand(1);
	llvm::Value * idxVal = insert->getOperand(2);

	const VariableDesc * desc = locals.lookUp(insert);
	assert(desc && "undeclared instruction");
	std::string descStr = desc->name;

	//const llvm::VectorType * vecType = llvm::cast<llvm::VectorType>(vec->getType());
	//const llvm::Type * elemType = vecType->getElementType();


	const VariableDesc * vecDesc = locals.lookUp(vec);
	std::string vecStr;
	if (vecDesc) {
		vecStr = vecDesc->name;
	} else if (! llvm::isa<llvm::UndefValue>(vec)) {
		assert(llvm::isa<llvm::Constant>(vec) && "non constant was not a declared variable");
		vecStr = getLiteral(llvm::cast<llvm::Constant>(vec));
	}

	const VariableDesc * valueDesc = locals.lookUp(value);
	std::string valueStr;
	if (valueDesc) {
		valueStr = valueDesc->name;
	} else {
		assert(llvm::isa<llvm::Constant>(value) && "non constant was not a declared variable");
		valueStr = getLiteral(llvm::cast<llvm::Constant>(value));
	}

	if (! llvm::isa<llvm::UndefValue>(vec)) {
		putLine( descStr + " = " + vecStr + ";" );
	}


	uint64_t index;
	if (! evaluateInt(idxVal, index)) {
		Log::fail(insert, "non-static parameter access");
	}

	putLine( descStr + ".s" + hexstr(index) + " = " + valueStr + ";" );

	/*int width = vecType->getNumElements();

	//build the string

	std::string result = "(" + getScalarType(elemType) + str<int>(width) + ")(";

	result += vecStr + ".s";
	for(int i = 0; i < index; ++i) {
		result += hexstr(i);
	}

	result += ", " + valueStr + ", " + vecStr + ".s";

	for(int i = index + 1; i < width; ++i) {
		result += hexstr(i);
	}

	result += ")";

	return result;*/
}


/*
* write a single instruction or atomic value as isolated expression
*/
std::string OCLWriter::getInstructionAsExpression(llvm::Instruction * inst, IdentifierScope & locals)
{
   //catch loads as they need a dereferenced operand
	if (llvm::isa<llvm::LoadInst>(inst)) {
		llvm::LoadInst * load = llvm::cast<llvm::LoadInst>(inst);
		llvm::Value * pointer = load->getOperand(0);

		if (load->isVolatile()) {
			return "*(" + getVolatilePointerTo(pointer, locals) + ")";
		} else {
			return getReferenceTo(pointer, locals);
		}


	//interpret this PHINode as a variable assignment
	} else if (llvm::isa<llvm::PHINode>(inst)) {
		llvm::PHINode * phi = llvm::cast<llvm::PHINode>(inst);
		const VariableDesc * commonDesc = locals.lookUp(phi->getIncomingValue(0));
		return commonDesc->name;

	//function call
	} else if (llvm::isa<llvm::CallInst>(inst)) {
		llvm::CallInst * call = llvm::cast<llvm::CallInst>(inst);
		llvm::Function * callee = call->getCalledFunction();


		// OpenCL special treatment of barrier/memfence instructions
		if (callee->getName() == "barrier" ||
			callee->getName() == "mem_fence")
		{
			std::string calleeName = callee->getName();

			llvm::Value * arg = call->getArgOperand(0);
			std::string enumStr;
			if (! evaluateEnum_MemFence(arg, enumStr)) {
				Log::fail(call, "expected enum value");
			}

			std::string result = calleeName + "(" + enumStr + ")";
			return result;


		} else { //regular intrinsic
		#ifdef DEBUG
			std::cerr << "intrinsic call " << callee->getName().str() << "\n";
#endif
			std::vector<std::string> operands;
		    const ArgList & argList = callee->getArgumentList();
		    ArgList::const_iterator argStart = argList.begin();
		    ArgList::const_iterator itArg = argStart;

		  for(uint argIdx = 0; argIdx < call->getNumArgOperands(); ++argIdx, ++itArg)
		  {
			 llvm::Value * op = call->getArgOperand(argIdx);
			 const VariableDesc * opDesc = locals.lookUp(op);

			 bool isByValue = itArg->hasByValAttr();

			 if (opDesc) {
				 //global variables and allocas are dereferenced by their name
				 if (opDesc->isAlloca || llvm::isa<llvm::GlobalVariable>(op)) {
					 if (isByValue)
						 operands.push_back(opDesc->name);
					 else
						 operands.push_back("&" + opDesc->name);

			     // all other pointer values need explicit dereferencing
				 } else {
					 if (isByValue)
						 operands.push_back("*(" + opDesc->name + ")");
					 else
						 operands.push_back(opDesc->name);
				 }
			 } else {
				std::string operandStr;

				//### operand instruction without parent block
				if (llvm::isa<llvm::Instruction>(op))
					operandStr = getInstructionAsExpression(llvm::cast<llvm::Instruction>(op), locals);
				else //### non-runtime dependent value
					operandStr = getNonInstruction(op, locals);

				operands.push_back(operandStr);
			 }
		  }

		  return getInstruction(inst, operands);
		}

	//# shuffle vector expression
	} else if (llvm::isa<llvm::ShuffleVectorInst>(inst)) {
		return getShuffleInstruction(llvm::cast<llvm::ShuffleVectorInst>(inst), locals);

	//# extract element expression
	} else if (llvm::isa<llvm::ExtractElementInst>(inst)) {
		return getExtractElementInstruction(llvm::cast<llvm::ExtractElementInst>(inst), locals);

	//# dynamic expression (use deref and then obtain element ptr)
	} else if (llvm::isa<llvm::GetElementPtrInst>(inst)) {
		return getPointerTo(inst, locals);

	} else if (llvm::isa<llvm::ExtractValueInst>(inst)) {
#ifdef DEBUG
		std::cerr << "is extract value(!)\n";
#endif
		return getReferenceTo(inst, locals);
	}

	assert(!llvm::isa<llvm::InsertElementInst>(inst) && "insert element instructions is translated into multiple statements (use writeInsertElementInstruction)");


	/*
	 * generic call-style instruction
	 */
	std::vector<std::string> operands;
  for(uint opIdx = 0; opIdx < inst->getNumOperands(); ++opIdx)
  {
     llvm::Value * op = inst->getOperand(opIdx);
     const VariableDesc * opDesc = locals.lookUp(op);

     if (opDesc) {
    	 //assigned value
    	 if (opDesc->isAlloca) {
    		 operands.push_back("&" + opDesc->name);
    	 } else {
    		 operands.push_back(opDesc->name);
    	 }

     } else { //inline entity
    	 std::string operandStr;

		//### operand instruction without parent block
		if (llvm::isa<llvm::Instruction>(op))
			operandStr = getInstructionAsExpression(llvm::cast<llvm::Instruction>(op), locals);
		else //### non-runtime dependent value
			operandStr = getNonInstruction(op, locals);

		operands.push_back(operandStr);
     }
  }

  return getInstruction(inst, operands);
}

/*
* write a complex expression made up of elements from valueBlock, starting from root, writing all included insts to @oExpressionInsts
*/
std::string OCLWriter::getComplexExpression(llvm::BasicBlock * valueBlock, llvm::Value * root, IdentifierScope & locals, InstructionSet * oExpressionInsts)
{
 // PHI-Nodes cant be part of a single expression
if (llvm::isa<llvm::Instruction>(root) &&
        ! llvm::isa<llvm::PHINode>(root) &&
        ! llvm::isa<llvm::GetElementPtrInst>(root))
  {
     llvm::Instruction * inst = llvm::cast<llvm::Instruction>(root);
     if (inst->getParent() == valueBlock) { //recursively decompose into operator queries

    	 if (oExpressionInsts)
    		 oExpressionInsts->insert(inst);

        std::vector<std::string> operands;
        for(uint opIdx = 0; opIdx < inst->getNumOperands(); ++opIdx)
        {
           llvm::Value * op = inst->getOperand(opIdx);

           operands.push_back(getComplexExpression(valueBlock, op, locals, oExpressionInsts));
        }

        return getInstruction(inst, operands);
     } else {
        const VariableDesc * desc = locals.lookUp(inst);
        return desc->name;
     }
  } else { //non-instruction value
	 return getNonInstruction(root, locals);
  }
}


/*
* writes a generic function header for utility functions and the default signature for the shade func
*/
 void OCLWriter::writeFunctionHeader(llvm::Function * func, IdentifierScope * locals)
{
  putLine( getFunctionHeader(func, locals));
}

 void OCLWriter::writeInstruction(const VariableDesc * desc, llvm::Instruction * inst, IdentifierScope & locals)
{
#ifdef DEBUG
		std::cerr << "writeInstruction:: var=" << (desc ? desc->name : "NULL") << std::endl;
		std::cerr << '\t';
		inst->dump();
#endif
	//dont write instructions consumed by their value users
	if (llvm::isa<llvm::GetElementPtrInst>(inst) ||
                llvm::isa<llvm::AllocaInst>(inst))
		return;

	// ### PHI-Node
    if (llvm::isa<llvm::PHINode>(inst))
    {
    	const VariableDesc * phiDesc = locals.lookUp(inst);
    	writeAssignRaw(phiDesc->name, phiDesc->name + "_in");
    	return;
    }

    //### Store instruction
	if (llvm::isa<llvm::StoreInst>(inst)) {
		llvm::StoreInst * store = llvm::cast<llvm::StoreInst>(inst);
		llvm::Value * pointer = store->getOperand(1);
		const VariableDesc * valueDesc = locals.lookUp(store->getOperand(0));

		llvm::Value * op = store->getOperand(0);

		std::string srcString;
		//fetch reference of pointer target to obtain the address value
		if (llvm::isa<llvm::PointerType>(op->getType())) {
			srcString = getPointerTo(op, locals);

		//pass values by identifier
		} else if (valueDesc) {
			srcString = valueDesc->name;

		} else { //otw
			assert(llvm::isa<llvm::Constant>(op));
			srcString = getLiteral(llvm::cast<llvm::Constant>(op));
		}

		//decode the GEP and store the value
		if (store->isVolatile()) {
			std::string ptr = getVolatilePointerTo(pointer, locals);
			std::string name = "*(" + ptr + ")";
			writeAssignRaw(name, srcString);
		} else {
			std::string name = getReferenceTo(pointer, locals);
			writeAssignRaw(name, srcString);
		}

	//### InsertElement Instruction
	} else if (llvm::isa<llvm::InsertElementInst>(inst)) {
		writeInsertElementInstruction(llvm::cast<llvm::InsertElementInst>(inst), locals);

	//### InsertValue Instruction
	} else if (llvm::isa<llvm::InsertValueInst>(inst)) {
		writeInsertValueInstruction(llvm::cast<llvm::InsertValueInst>(inst), locals);

	//### assigning instruction
	} else if (desc) {
		writeAssignRaw(desc->name, getInstructionAsExpression(inst, locals));

	//### void/discarded result instruction
	} else {
		std::string instStr = getInstructionAsExpression(inst, locals);
		if (instStr != "") {
			putLine( instStr + ';');
		}
	}
}

void OCLWriter::writeIf(const llvm::Value * condition, bool negate, IdentifierScope & locals)
{
	const VariableDesc * condVar = locals.lookUp(condition);

	std::string condStr;

	if (condVar) {
		condStr = condVar->name;

	} else if (llvm::isa<llvm::ConstantInt>(condition)) {
		const llvm::ConstantInt * constInt = llvm::cast<llvm::ConstantInt>(condition);
		condStr = constInt->isZero() ? "false" : "true";

	} else {
		assert(false && "controlling expression must be a literal or have a bound identifier");
	}

	if (! negate)
		putLine( "if (" + condStr + ")" );
	else
		putLine( "if (! " + condStr + ")" );
}

 void OCLWriter::writeElse()
{
	putLine( "else" );
}

 void OCLWriter::writeLoopContinue()
{
	putLine("continue;");
}

 void OCLWriter::writeLoopBreak()
{
	putLine( "break;" );
}

 void OCLWriter::writeDo()
{
  putLine( "do" );
}

void OCLWriter::writeAssign(const VariableDesc & dest, const VariableDesc & src)
{
#ifdef DEBUG
	std::cerr << "writeAssign:: enforcing assignment to " << dest.name << std::endl;
#endif
	writeAssignRaw(dest.name, src.name);
}

void OCLWriter::writeAssignRaw(const std::string & destStr, llvm::Value * val, IdentifierScope & locals)
{
	const VariableDesc * srcDesc = locals.lookUp(val);
	std::string srcText;

	if (srcDesc) {
		srcText = srcDesc->name;
	} else if (llvm::isa<llvm::GetElementPtrInst>(val)) {
		srcText = getPointerTo(val, locals);
	} else if (llvm::isa<llvm::Constant>(val)) {
		srcText = getConstant(llvm::cast<llvm::Constant>(val), locals);
	} else {
		Log::fail(val, "source values of this kind are not covered in writeAssignRaw (TODO)");
	}

	putLine(destStr + " = " + srcText + ";");
}

void OCLWriter::writeAssignRaw(const std::string & dest, const std::string & src)
{
	putLine(dest + " = " + src + ";");
}

/*
 * write a while for a post<checked loop
 */
void OCLWriter::writePostcheckedWhile(llvm::BranchInst * branchInst, IdentifierScope & locals, bool negate)
{
   llvm::Value * loopCond = branchInst->getCondition();

   const VariableDesc * desc = locals.lookUp(loopCond);

   assert(desc && "was not mapped");

    std::string expr = desc->name;

    if (negate)
    	putLine( "while (! " + expr + ");" );
    else
    	putLine( "while (" + expr + ");" );
}

/*
 * write a while for a postchecked loop, if oExpressionInsts != NULL dont write, but put all consumed instructions in the set
 */
 void OCLWriter::writePrecheckedWhile(llvm::BranchInst * branchInst, IdentifierScope & locals, bool negate, InstructionSet * oExpressionInsts)
{
  llvm::Value * loopCond = branchInst->getCondition();

  std::string expr = getComplexExpression(branchInst->getParent(), loopCond, locals, oExpressionInsts);

  if (! oExpressionInsts) {
	  if (negate)
		  putLine( "while (! " + expr + ")" );
	  else
		  putLine( "while (" + expr + ")" );
  }
}

 void OCLWriter::writeInfiniteLoopBegin()
{
   putLine ( "while(true)" );
}

 void OCLWriter::writeInfiniteLoopEnd()
{
   putLine( "" );
}

 void OCLWriter::writeReturnInst(llvm::ReturnInst * retInst, IdentifierScope & locals)
	{
   if (retInst->getNumOperands() > 0) //return value
   {
		const VariableDesc * desc = locals.lookUp(retInst->getOperand(0));

		if (desc) {
			putLine( "return " +  desc->name + ";" );
		} else {
			llvm::Value * retVal = retInst->getReturnValue();

			if (! llvm::isa<llvm::UndefValue>(retVal))
			{
				putLine( "return " + getNonInstruction(retInst->getReturnValue(), locals) + ";" );
			} else {
				Log::warn(retInst, "skipping return for the returned value is undefined!");
			}
		}

   } else { ///void return
	   putLine( "return;" );
   }
	}

/*
* writes a generic struct type declaration to the fragment shader
*/
 std::string OCLWriter::getStructTypeDeclaration(const std::string & structName, const llvm::StructType * structType)
{
   std::string res =  "struct " + structName + "\n";
   res +=  "{\n";

   for(uint i = 0; i < structType->getNumElements(); ++i)
   {
	   const llvm::Type * elementType = structType->getElementType(i);

	   std::string memberName = "x" + str<int>(i);
	   std::string memberStr = buildDeclaration(memberName, elementType);

	   res += INDENTATION_STRING + memberStr + ";\n";
   }

   res += "};\n";
   return res;
}

 void OCLWriter::writeFunctionPrologue(llvm::Function * func, IdentifierScope & locals)
{
   typedef llvm::Function::ArgumentListType ArgList;
   const ArgList & argList = func->getArgumentList();

   ConstValueSet arguments;
   for(ArgList::const_iterator arg = argList.begin(); arg != argList.end(); ++arg)
   {
	   arguments.insert(arg);
   }

   for(ConstVariableMap::iterator itVar = locals.identifiers.begin();
	   itVar != locals.identifiers.end();
	   ++itVar)
   {
	   VariableDesc & desc = itVar->second;

#ifdef DEBUG
	   itVar->first->print(llvm::outs()); std::cout << " - > " << itVar->second.name << std::endl;
#endif

	   if (arguments.find(itVar->first) == arguments.end()) {
		   if (llvm::isa<llvm::PHINode>(itVar->first))
		   {
			   VariableDesc inputDesc;
			   inputDesc.name = desc.name + "_in";
			   inputDesc.type = desc.type;
			   writeVariableDeclaration(inputDesc);
		   }
		   writeVariableDeclaration(desc);
	   }
   }

   //dump local storage declarations to kernel functions
   if (modInfo.isKernelFunction(func))
   {
		for (llvm::Module::global_iterator global = func->getParent()->global_begin(),
         end = func->getParent()->global_end();
         global != end; ++global) {
			std::string varName = global->getName().str();
			const llvm::PointerType * gvType = global->getType();
			const llvm::Type * contentType = gvType->getElementType();

			//spill dynamic local globals in kernel space
			if (gvType->getAddressSpace() == SPACE_LOCAL && !global->isConstant()) {
				std::string spaceName = getAddressSpaceName(SPACE_LOCAL);

        // Check kernel name agains current function.
        if (! usedInFunction(func, &*global))
          continue;

				std::string declareStr = buildDeclaration(getLocalVariableName(varName),
                                                  contentType) + ";";
				putLine(spaceName + " " + declareStr );
			}
		}
   }
   writeLineBreak();
}

/*
* dumps a generic vertex shader and all type&argument defs for the frag shader
*/
OCLWriter::OCLWriter(ModuleInfo & _modInfo, PlatformInfo & _platform) :
		modInfo(reinterpret_cast<OCLModuleInfo&>(_modInfo)),
		platform(_platform)
	{
		llvm::Module * mod = modInfo.getModule();

		//### dump extensions ###
		if (modInfo.requiresDoubleType())
			putLine("#pragma OPENCL EXTENSION cl_khr_fp64: enable");

		putLine("");
				                
                llvm::TypeFinder SrcStructTypes;
                SrcStructTypes.run(*mod, true);
                StructTypeVector types(SrcStructTypes.begin(),
                                                 SrcStructTypes.end());

		//### print type declarations ###
		{
			//### sort dependent types for forward declaration ###
			{
				TypeSet undeclaredTypes;

				for (StructTypeVector::iterator itType = types.begin(); itType != types.end(); ++itType) {
					undeclaredTypes.insert(*itType);
				}

				for (StructTypeVector::iterator itType = types.begin(); itType != types.end();) {

					const llvm::StructType * type = *itType;
					const std::string name = (*itType)->getName().str();

					if (
							undeclaredTypes.find(type) == undeclaredTypes.end() || //already declared
							containsType(type, undeclaredTypes))
					{
						++itType;
					} else {

						//write declaration
						std::string structStr = getStructTypeDeclaration(name, llvm::cast<const llvm::StructType>(type));
						put( structStr );

						undeclaredTypes.erase(type);
						itType = types.begin();
					}
				}
			}
		}

		putLine( "" );

		//## spill globals
		for (llvm::Module::global_iterator global = mod->global_begin(); global != mod->global_end(); ++global)
		{
			if (llvm::isa<llvm::GlobalVariable>(global)) {
				llvm::GlobalVariable * var = llvm::cast<llvm::GlobalVariable>(global);
				const llvm::PointerType * varType = var->getType();
				const llvm::Type * contentType = varType->getElementType();
				std::string varName = var->getName().str();

				uint space = varType->getAddressSpace();

				if (var->isConstant()) { // __constant
					std::string initStr = getLiteral(var->getInitializer());
					std::string declareStr = buildDeclaration(varName, contentType) + " = " + initStr + ";";
					putLine( "__constant " + declareStr);

				} else if (space == SPACE_GLOBAL) { //global variable with external initialization (keep the pointer)
					std::string spaceName = getAddressSpaceName(space);
					std::string declareStr = buildDeclaration(varName, varType) + ";";

					putLine(spaceName + " " + declareStr );
				} else if (space == SPACE_LOCAL) { //work group global variable with initialization (declare for content type)
				/*	std::string spaceName = getAddressSpaceName(space);
					std::string declareStr = buildDeclaration(varName, contentType) + ";";

					putLine(spaceName + " " + declareStr );*/
					//will declare this variable in any using kernel function

				} else {
					Log::warn(var, "discarding global variable declaration");
				}
			}
		}

		putLine( "" );

		//## spill function declarations
		for (llvm::Module::iterator func = mod->begin(); func != mod->end(); ++func)
		{
			if (! platform.implements(func))
				putLine(getFunctionHeader(func) + ";");
		}

		putLine( "" );

#ifdef DEBUG
		std::cerr << "completed OCLWriter ctor\n";
#endif
	}

OCLWriter::OCLWriter(OCLWriter & writer) :
		modInfo(writer.modInfo),
		platform(writer.platform)
{}


template class ResourceGuard<AddressIterator>;


/*
 * BlockWriter (indents all writes and embraces them in curly brackets)
 */
 void OCLBlockWriter::put(std::string text)
{
	parent.put(INDENTATION_STRING + text);
}

OCLBlockWriter::OCLBlockWriter(OCLWriter & _parent) :
	OCLWriter(_parent),
	parent(_parent)
{
	parent.putLine("{");
}

 OCLBlockWriter::~OCLBlockWriter()
{
	parent.putLine("}");
}



/*
 * PassThrough writer
 */
OCLPassThroughWriter::OCLPassThroughWriter(OCLWriter & _parent) :
	OCLWriter(_parent),
	parent(_parent)
{}

void OCLPassThroughWriter::put(std::string msg)
{
	parent.put(msg);
}

/*
 * Multi writer
 */
OCLMultiWriter::OCLMultiWriter(OCLWriter & _first, OCLWriter & _second) :
		OCLWriter(_first),
		first(_first), second(_second)
{}

void OCLMultiWriter::put(std::string msg)
{
	first.put(msg);
	second.put(msg);
}

}
