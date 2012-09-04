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
 * preparator.h
 *
 *  Created on: 12.02.2010
 *      Author: Simon Moll
 */

#ifndef PREPARATOR_HPP_
#define PREPARATOR_HPP_

#include <axtor/config.h>

#include <llvm/PassManager.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/Dominators.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/PassManagers.h>
#include <llvm/Module.h>
#include <llvm/Pass.h>
#include <llvm/ADT/APFloat.h>
#include <llvm/Instructions.h>
#include <llvm/Instruction.h>
#include <llvm/Analysis/Passes.h>

#include <axtor/util/stringutil.h>
#include <axtor/util/llvmShortCuts.h>

#include <axtor/writer/SyntaxWriter.h>
#include <axtor/CommonTypes.h>

#include <axtor/pass/ExitUnificationPass.h>

/*
 * This pass assigns names to all types and instructions in the module. Opaque Type names are preserved.
 * Also it replaces constants in instructions by uses of fake assignment instructions
 */
namespace axtor
{
	class Preparator : public llvm::ModulePass
	{
	private:
		/*
		 * names all unnamed values in a function
		 */
		void nameAllInstructions(llvm::Function * func);

		void nameAllInstructions(llvm::Module * mod);

		/*
		 * convert all constant arguments of PHI-Nodes to instructions in the entry block
		 */

		llvm::Instruction * createAssignment(llvm::Value * val, llvm::Instruction * beforeInst);

		llvm::Instruction * createAssignment(llvm::Value * val, llvm::BasicBlock * block);

		void transformStoreConstants(llvm::StoreInst * store);

		struct AssignmentBridge
		{
			llvm::BasicBlock * bridge;
			llvm::Instruction * value;

			AssignmentBridge(llvm::BasicBlock * _bridge, llvm::Instruction * _value);
		};

		AssignmentBridge insertBridgeForBlock(llvm::Instruction * inst, llvm::BasicBlock * branchTarget);

		/*
		 * transforms PHINodes into a standardized format where
		 * no phi uses neither PHI-Nodes nor constants immediately
		 */
		void transformPHINode(llvm::PHINode * phi);

		/*
		 * calls specific instruction transform methods
		 * (for stores and PHIs so far)
		 */
		void transformInstruction(llvm::Instruction * inst);

		/*
		 * transforms all instructions occuring in a module if necessary
		 */
		void transformInstArguments(llvm::Module * mod);

		/*
		 * inserts all constituent struct types of @type into @types
		 */
		void insertIfStruct(TypeSet & types, const llvm::Type * type);

		/*
		 * give a generic name to all structs in @symTable
		 */
		void cleanStructNames(llvm::Module & M);


		/*
		 * renames all globals with internal linkage
		 */
		void cleanInternalGlobalNames(llvm::Module & M);


		/*
		 * sort TypeSymbols
		 *
		 * (avoid forward declarations)
		 */
		//void sortTypeSymbolTable(llvm::TypeSymbolTable & symTable);

	public:
		static char ID;

			virtual const char * getPassName() const;

			virtual void getAnalysisUsage(llvm::AnalysisUsage & usage) const;

			Preparator();

			virtual ~Preparator();

			virtual bool runOnModule(llvm::Module& M);
		};
}


#endif /* PREPARATOR_HPP_ */
