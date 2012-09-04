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
 * SimpleUnswitchPass.h
 *
 *  Created on: 16.03.2011
 *      Author: Simon Moll
 */


#ifndef SIMPLEUNSWITCHPASS_HPP_
#define SIMPLEUNSWITCHPASS_HPP_

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

#include <axtor/util/llvmShortCuts.h>
#include <axtor/CommonTypes.h>


namespace axtor {
	/*
	 * replaces Switch- terminators with a cascade of 2-way branches
	 */
	class SimpleUnswitchPass : public llvm::ModulePass
	{
		bool runOnFunction(llvm::Function * func);

		void processSwitch(llvm::Function * func, llvm::BasicBlock * switchBlock);

	public:
		static char ID;

		virtual const char * getPassName() const;

		virtual void getAnalysisUsage(llvm::AnalysisUsage & usage) const;

		SimpleUnswitchPass();

		virtual ~SimpleUnswitchPass();

		virtual bool runOnModule(llvm::Module& M);
	};

}

#endif /* SIMPLEUNSWITCHPASS_HPP_ */
