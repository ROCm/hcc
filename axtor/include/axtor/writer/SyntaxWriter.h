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
 * SyntaxWriter.h
 *
 *  Created on: 19.03.2010
 *      Author: Simon Moll
 */

#ifndef SYNTAXWRITER_HPP_
#define SYNTAXWRITER_HPP_

#include <axtor/CommonTypes.h>
#include <axtor/metainfo/ModuleInfo.h>

namespace axtor {

/*
 * SyntaxWriter - interface (used by the AST-extraction core during translation)
 */
struct SyntaxWriter
{
	virtual ~SyntaxWriter() {}

	virtual void dump() = 0;
	virtual void print(std::ostream & out) = 0;

	//### Prologues ###
	virtual void writeFunctionPrologue(llvm::Function * func, IdentifierScope & locals)=0; //should declare all variables in locals

	//### Declarations ###
	virtual void writeVariableDeclaration(const VariableDesc & desc) = 0;
	virtual void writeFunctionDeclaration(llvm::Function * func, IdentifierScope * locals = NULL) = 0;

	virtual void writeFunctionHeader(llvm::Function * func, IdentifierScope * locals = NULL) = 0;

	//### Control Flow Elements ##
	virtual void writeIf(const llvm::Value * condition, bool negateCondition, IdentifierScope & locals)=0;
	virtual void writeElse()=0;
	virtual void writeLoopContinue()=0;
	virtual void writeLoopBreak()=0;
	virtual void writeDo()=0;


	//### Loops ###
	virtual void writeInfiniteLoopBegin()=0;
	virtual void writeInfiniteLoopEnd()=0;

	virtual void writePostcheckedWhile(llvm::BranchInst * branchInst, IdentifierScope & locals, bool negate)=0;

	virtual void writePrecheckedWhile(llvm::BranchInst * branchInst, IdentifierScope & locals, bool negate, InstructionSet * oExpressionInsts)=0;

	void writePrecheckedWhile(llvm::BranchInst * branchInst, IdentifierScope & locals, bool negate)
	{
	   return writePrecheckedWhile(branchInst, locals, negate, NULL);
	}

	//### Instructions ###
	virtual void writeAssignRaw(const std::string & destName, llvm::Value * val, IdentifierScope & locals)=0;
	virtual void writeAssign(const VariableDesc & dest, const VariableDesc & src)=0;
	virtual void writeAssignRaw(const std::string & dest, const std::string & src)=0;
	virtual void writeReturnInst(llvm::ReturnInst * retInst, IdentifierScope & locals)=0;
	virtual void writeInstruction(const VariableDesc * desc, llvm::Instruction * inst, IdentifierScope & locals) = 0;
};

}

#endif /* SYNTAXWRITER_HPP_ */
