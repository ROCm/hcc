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
 * LOOPBRANCHSEPARATIONPASS.h
 *
 *  Created on: 19.02.2011
 *      Author: Simon Moll
 */

#ifndef LOOPBRANCHSEPARATIONPASS_HPP_
#define LOOPBRANCHSEPARATIONPASS_HPP_

#include <axtor/config.h>

#include <llvm/Module.h>
#include <llvm/Pass.h>
#include <llvm/BasicBlock.h>
#include <llvm/Analysis/Dominators.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include <axtor/console/CompilerLog.h>
#include <axtor/util/llvmShortCuts.h>
#include <axtor/util/llvmDomination.h>
#include <axtor/CommonTypes.h>
#include <axtor/pass/TargetProvider.h>



namespace axtor {

/*
 * Detaches loop branches from 2-way conditionals to simplify the AST-extraction
 */
class LoopBranchSeparationPass : public llvm::ModulePass
{
	/*
	 * breaks the edge and inserts adapts the PHI-nodes
	 */
	llvm::BasicBlock * breakSpecialEdge(llvm::Function * func, llvm::BasicBlock * srcBlock, llvm::BasicBlock * targetBlock, llvm::Function::iterator insertBefore);

	bool runOnFunction(llvm::Function & func);
public:
	static char ID;

	LoopBranchSeparationPass();

	virtual void getAnalysisUsage(llvm::AnalysisUsage & usage) const;

	bool runOnModule(llvm::Module & mod);

	virtual const char * getPassName() const;
};

}


#endif /* LOOPBRANCHSEPARATIONPASS_HPP_ */
