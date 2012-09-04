/*
 * PassThroughWriter.cpp
 *
 *  Created on: 03.05.2010
 *      Author: gnarf
 */

#include<axtor/writer/PassThroughWriter.h>

namespace axtor {

PassThroughWriter::PassThroughWriter(SyntaxWriter & _writer) :
	writer(_writer)
{}

PassThroughWriter::~PassThroughWriter()
{}

void PassThroughWriter::dump()
{
	writer.dump();
}

void PassThroughWriter::print(std::ostream & out)
{
	writer.print(out);
}

	//### Prologues ###
void PassThroughWriter::writeFunctionPrologue(llvm::Function * func, IdentifierScope & locals)
{
	writer.writeFunctionPrologue(func, locals);
}

	//### Declarations ###
void PassThroughWriter::writeVariableDeclaration(const VariableDesc & desc)
{
	writer.writeVariableDeclaration(desc);
}
void PassThroughWriter::writeFunctionDeclaration(llvm::Function * func, IdentifierScope * locals)
{
//	writer.writeFunctionDeclaration(func, locals);
}

void PassThroughWriter::writeFunctionHeader(llvm::Function * func, IdentifierScope * locals)
{
	writer.writeFunctionHeader(func, locals);
}

	//### Control Flow Elements ##
void PassThroughWriter::writeIf(const llvm::Value * value, bool negateCondition, IdentifierScope & scope)
{
	writer.writeIf(value, negateCondition, scope);
}

void PassThroughWriter::writeElse()
{
	writer.writeElse();
}

void PassThroughWriter::writeLoopContinue()
{
	writer.writeLoopContinue();
}

void PassThroughWriter::writeLoopBreak()
{
	writer.writeLoopBreak();
}

void PassThroughWriter::writeDo()
{
	writer.writeDo();
}


	//### Loops ###
void PassThroughWriter::writeInfiniteLoopBegin()
{
	writer.writeInfiniteLoopBegin();
}

void PassThroughWriter::writeInfiniteLoopEnd()
{
	writer.writeInfiniteLoopEnd();
}

void PassThroughWriter::writePostcheckedWhile(llvm::BranchInst * branchInst, IdentifierScope & locals, bool negate)
{
	writer.writePostcheckedWhile(branchInst, locals, negate);
}

void PassThroughWriter::writePrecheckedWhile(llvm::BranchInst * branchInst, IdentifierScope & locals, bool negate, InstructionSet * oExpressionInsts)
{
	writer.writePrecheckedWhile(branchInst, locals, negate, oExpressionInsts);
}

//### Instructions ###
void PassThroughWriter::writeAssign(const VariableDesc & dest, const VariableDesc & src)
{
	writer.writeAssign(dest, src);
}

void PassThroughWriter::writeReturnInst(llvm::ReturnInst * retInst, IdentifierScope & locals)
{
	writer.writeReturnInst(retInst, locals);
}

void PassThroughWriter::writeInstruction(const VariableDesc * desc, llvm::Instruction * inst, IdentifierScope & locals)
{
	writer.writeInstruction(desc, inst, locals);
}

}
