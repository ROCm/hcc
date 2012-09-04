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
 * Serializer.h
 *
 *  Created on: 12.06.2010
 *      Author: Simon Moll
 */

#ifndef SERIALIZER_HPP_
#define SERIALIZER_HPP_

#include <axtor/config.h>

#include <llvm/Pass.h>
#include <llvm/PassSupport.h>

#include <axtor/backend/AxtorBackend.h>
#include <axtor/CommonTypes.h>

#include <axtor/pass/RestructuringPass.h>

/*
 * This pass gets an AST from the ASTExtractor pass and serializes it using the backend specified by the TargetProvider pass
 */
namespace axtor
{
	class Serializer : public llvm::ModulePass
	{
	private:
		void runOnFunction(AxtorBackend & backend, SyntaxWriter * modWriter, IdentifierScope & globals, ast::FunctionNode * funcNode);

	public:
		static char ID;
		Serializer();


		/*
		 * replace all resolved PHI-nodes by assignments
		 */
		void processBranch(SyntaxWriter * writer, llvm::BasicBlock * source, llvm::BasicBlock * target, IdentifierScope & locals);

		/*
		 * writes all instructions of a @bb except the terminator except those contained in @supresssedInsts
		 */
		void writeBlockInstructions(SyntaxWriter * writer, llvm::BasicBlock * bb, IdentifierScope & identifiers);

		/*
		 * creates identifier bindings for function arguments
		 */
		void createArgumentDeclarations(llvm::Function * func, ConstVariableMap & declares, std::set<llvm::Value*> & parameters);

		void getAnalysisUsage(llvm::AnalysisUsage & usage) const;
		bool runOnModule(llvm::Module & M);

		const char * getPassName() const;

		/*
		 * writes a node using the given backend and writer.
		 * @return unique exiting block (if any)
		 */
		llvm::BasicBlock * writeNode(AxtorBackend & backend, SyntaxWriter * writer, llvm::BasicBlock * previousBlock, llvm::BasicBlock * exitBlock, ast::ControlNode * node, IdentifierScope & locals, llvm::BasicBlock * loopHeader, llvm::BasicBlock * loopExit);
	};
}


#endif /* SERIALIZER_HPP_ */
