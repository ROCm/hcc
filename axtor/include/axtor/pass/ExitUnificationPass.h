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
 * ExitUnificationPass.h
 *
 *  Created on: 05.03.2010
 *      Author: Simon Moll
 */

#ifndef EXITUNIFICATIONPASS_HPP_
#define EXITUNIFICATIONPASS_HPP_

#include <axtor/config.h>

#include <llvm/Pass.h>
#include <llvm/Module.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/BasicBlock.h>
#include <llvm/Instructions.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/SmallVector.h>

#include <axtor/util/llvmShortCuts.h>
#include <axtor/util/llvmConstant.h>
#include <axtor/CommonTypes.h>
#include <axtor/pass/CNSPass.h>

namespace axtor {

/*
 * this pass enumerates all loop exits to the parent loop and pulls them out into a single branching exit block
 */
class ExitUnificationPass : public llvm::ModulePass
{

	/*
	 * generate a switch-like construct branching to elements of dest on their index
	 */
	void appendEnumerationSwitch(llvm::Value * val, std::vector<llvm::BasicBlock*> dests, llvm::BasicBlock * block);

	/*
	 * if this loop has multiple exits to the parent loop enumerate them and move the branches to a dedicated exit block
	 *
	 * @return true, if the loop has changed
	 */
	bool unifyLoopExits(llvm::Function & func, llvm::Loop * loop);

	bool runOnFunction(llvm::Function & func);

public:
	static char ID;

	ExitUnificationPass();

	virtual void getAnalysisUsage(llvm::AnalysisUsage & usage) const;

	bool runOnModule(llvm::Module & M);

	virtual const char * getPassName() const;
};

}



#endif /* EXITUNIFICATIONPASS_HPP_ */
