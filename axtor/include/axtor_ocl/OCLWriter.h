/*  Axtor - AST-Extractor for LLVM
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */
/*
 * OCLWriter.h
 *
 *  Created on: ?.2010
 *      Author: Simon Moll
 */

#ifndef OCLWRITER_HPP
#define OCLWRITER_HPP

#include <vector>
#include <set>
#include <string>

#include <llvm/Module.h>
#include <llvm/Value.h>
#include <llvm/Instruction.h>
#include <llvm/Function.h>
//#include <llvm/TypeSymbolTable.h>
#include <llvm/Module.h>
#include <llvm/Constants.h>
#include <llvm/Support/raw_ostream.h>

#include <axtor/writer/SyntaxWriter.h>
#include <axtor/CommonTypes.h>

#include <axtor/intrinsics/PlatformInfo.h>

#include <axtor/console/CompilerLog.h>

#include <axtor/util/WrappedOperation.h>
#include <axtor/util/llvmShortCuts.h>
#include <axtor/util/llvmConstant.h>
#include <axtor/util/stringutil.h>
#include <axtor/util/ResourceGuard.h>
#include <axtor/intrinsics/AddressIterator.h>

#include "OCLCommon.h"
#include "OCLModuleInfo.h"

#define INDENTATION_STRING "   "

namespace axtor {

class OCLBlockWriter;

/*
* Generic SlangWriter interface
*/
class OCLWriter : public SyntaxWriter
{
		friend class OCLBlockWriter;
		friend class OCLPassThroughWriter;
		friend class OCLMultiWriter;
private:
	OCLModuleInfo & modInfo;
	PlatformInfo & platform;

protected:
	virtual void put(std::string text);

	inline void putLine(std::string text);

	inline void putLineBreak();

	inline void put(char c);

	inline void putLine(char c);

private: //helper methods
	// consumes the first and all following cumulative values in a string to build an array dereferencing sum of indices "DATA[add0+add1+add2+add3]"
	std::string buildArraySubscript(std::string root, AddressIterator *& address, IdentifierScope & locals);

public:
	virtual void dump();

	virtual void print(std::ostream & out);

	/*
	 *  ##### Type System #####
	 */

	/*
	 * returns type symbols for default scalar types
	 */
	std::string getScalarType(const llvm::Type * type, bool asVectorElementType = false);

	std::string getAddressSpaceName(uint space);
  std::string getLocalVariableName(const std::string &variableFullName);
  std::string getLocalVariableKernel(const std::string &variableFullName);

	/*
	 * generates a type name for @type
	 * if this is a pointer type, operate on its element type instead
	 */
	std::string getType(const llvm::Type * type);

	/*
	 * build a C-style declaration for @root of type @type
	 */
	std::string buildDeclaration(std::string root, const llvm::Type * type);


	/*
	* ##### DECLARATIONS / OPERATORS & INSTRUCTIONS ######
	 */

  	/*
    * writes a generic function header and declares the arguments as mapped by @locals
    */
	std::string getFunctionHeader(llvm::Function * func, IdentifierScope * locals);

	std::string getFunctionHeader(llvm::Function * func);

	virtual void writeLineBreak();

	virtual void writeVariableDeclaration(const VariableDesc & desc);

	virtual void writeFunctionDeclaration(llvm::Function * func, IdentifierScope * locals = NULL);

	/*
	 * default C Style operators (returns if operand casting is required)
	 */
   std::string getOperatorToken(const WrappedOperation & op, bool & isSigned);

   /*
   * returns the string representation of a operator using @operands as operand literals
   */
   std::string getInstruction(llvm::Instruction * inst, std::vector<std::string> operands);

   /*
    * returns the string representation of a constant
    */
   std::string getConstant(llvm::Constant * constant, IdentifierScope & locals);

   /*
  * returns the string representation of a operator using @operands as operand literals
  */
   std::string getOperation(const WrappedOperation & operation, std::vector<std::string> operands);

   /*
    * returns the string referer to a value (designator for instructions/serialisation for constants)
    */
   std::string getReferer(llvm::Value * value, IdentifierScope & locals);

   typedef std::vector<llvm::Value*> ValueVector;

   /*
    * return a dereferencing string for the next type node of the object using address
    */
	std::string dereferenceContainer(std::string root, const llvm::Type * type, AddressIterator *& address, IdentifierScope & locals, const llvm::Type *& oElementType, uint addressSpace);

	// auxiliary functions for obtaining dereferencing or pointer to strings
	std::string getPointerTo(llvm::Value * val, IdentifierScope & locals, const std::string * rootName = 0);
	std::string getVolatilePointerTo(llvm::Value * val, IdentifierScope & locals, const std::string * rootName = 0);
	std::string getReferenceTo(llvm::Value * val, IdentifierScope & locals, const std::string * rootName = 0);
	/*
	 * return a name representing a dereferenced pointer
	*/
	std::string unwindPointer(llvm::Value * val, IdentifierScope & locals, bool & oDereferenced, const std::string * rootName = 0);

	std::string getAllNullLiteral(const llvm::Type * type);
	/*
	* return the string representation of a constant
	*/
   std::string getLiteral(llvm::Constant * val);

   /*
	* tries to create a literal string it @val does not have a variable
	*/
   std::string getValueToken(llvm::Value * val, IdentifierScope & locals);

   /*
   * returns the string representation of a non-instruction value
   */
   std::string getNonInstruction(llvm::Value * value, IdentifierScope & locals);

   /*
   * returns the string representation of a ShuffleInstruction
   */
	std::string getShuffleInstruction(llvm::ShuffleVectorInst * shuffle, IdentifierScope & locals);

	/*
	 * returns the string representation of an ExtractElementInstruction
	 */
	std::string getExtractElementInstruction(llvm::ExtractElementInst * extract, IdentifierScope & locals);

	/*
	 * returns the string representation of an InsertElement/ValueInstruction
	 * if the vector/compound value is defined this creates two instructions
	 */
	void writeInsertElementInstruction(llvm::InsertElementInst * insert, IdentifierScope & locals);
	void writeInsertValueInstruction(llvm::InsertValueInst * insert, IdentifierScope & locals);

	/*
   * write a single instruction or atomic value as isolated expression
   */
   std::string getInstructionAsExpression(llvm::Instruction * inst, IdentifierScope & locals);

   /*
   * write a complex expression made up of elements from valueBlock, starting from root, writing all included insts to @oExpressionInsts
   */
   std::string getComplexExpression(llvm::BasicBlock * valueBlock, llvm::Value * root, IdentifierScope & locals, InstructionSet * oExpressionInsts = NULL);

   /*
    * writes a generic function header for utility functions and the default signature for the shade func
    */
   virtual void writeFunctionHeader(llvm::Function * func, IdentifierScope * locals = NULL);

	virtual void writeInstruction(const VariableDesc * desc, llvm::Instruction * inst, IdentifierScope & locals);

	virtual void writeIf(const llvm::Value * condition, bool negateCondition, IdentifierScope & locals);

	virtual void writeElse();

	virtual void writeLoopContinue();

	virtual void writeLoopBreak();

	virtual void writeDo();

	//half-unchecked assign
	virtual void writeAssignRaw(const std::string & destName, llvm::Value * val, IdentifierScope & locals);

	virtual void writeAssign(const VariableDesc & desc, const VariableDesc & src);

	virtual void writeAssignRaw(const std::string & dest, const std::string & src);

	/*
	 * write a while for a post<checked loop
	 */
	void writePostcheckedWhile(llvm::BranchInst * branchInst, IdentifierScope & locals, bool negate);

	/*
	 * write a while for a postchecked loop, if oExpressionInsts != NULL dont write, but put all consumed instructions in the set
	 */
   virtual void writePrecheckedWhile(llvm::BranchInst * branchInst, IdentifierScope & locals, bool negate, InstructionSet * oExpressionInsts);

   virtual void writeInfiniteLoopBegin();

   virtual void writeInfiniteLoopEnd();

   virtual void writeReturnInst(llvm::ReturnInst * retInst, IdentifierScope & locals);

   /*
    * writes a generic struct type declaration to the module
    */
   virtual std::string getStructTypeDeclaration(const std::string & structName, const llvm::StructType * structType);

   virtual void writeFunctionPrologue(llvm::Function * func, IdentifierScope & locals);

   /*
    * spills all global declarations (variables && types)
    */
   OCLWriter(ModuleInfo & _modInfo, PlatformInfo & _platform);

protected:
   /*
    * used for nested writer creation
    */
   OCLWriter(OCLWriter & writer);
};

/*
 * forwards all writes to the parent object
 */
class OCLPassThroughWriter : public OCLWriter
{
	OCLWriter & parent;
public:
	OCLPassThroughWriter(OCLWriter & _parent);

protected:
	virtual void put(std::string text);
};

/*
 * writes the output to two streams
 */
class OCLMultiWriter : public OCLWriter
{
	OCLWriter first;
	OCLWriter second;

public:
	OCLMultiWriter(OCLWriter & _first, OCLWriter & _second);

protected:
	virtual void put(std::string text);
};

/*
 * BlockWriter (indents all writes and embraces them in curly brackets)
 */
class OCLBlockWriter : public OCLWriter
{
	OCLWriter & parent;

protected:
	virtual void put(std::string text);

public:

	OCLBlockWriter(OCLWriter & _parent);

	virtual ~OCLBlockWriter();
};

}

#endif
