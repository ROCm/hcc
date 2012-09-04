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
 * AnalysisStruct.h
 *
 *  Created on: 20.02.2011
 *      Author: Simon Moll
 */

#ifndef ANALYSISSTRUCT_HPP_
#define ANALYSISSTRUCT_HPP_


#include <llvm/PassManager.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/Dominators.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/PassManagers.h>
#include <llvm/Module.h>
#include <llvm/Pass.h>
#include <llvm/Instructions.h>
#include <llvm/Instruction.h>
#include <llvm/Analysis/Passes.h>
#include <llvm/ADT/APFloat.h>
#include <llvm/Support/raw_ostream.h>

#include <axtor/pass/AnalysisProvider.h>

namespace axtor{

	class AnalysisStruct
	{
		AnalysisProvider * provider;

		llvm::Function * func;
		llvm::LoopInfo * loopInfo;
		llvm::DominatorTree * domTree;
		llvm::PostDominatorTree * postDomTree;
	public:
		inline llvm::Function * getFunction() { return func; }
		inline llvm::LoopInfo & getLoopInfo() { return *loopInfo; }
		inline llvm::Loop * getLoopFor(const llvm::BasicBlock * block) { return loopInfo->getLoopFor(block); }

		inline llvm::DominatorTree & getDomTree() { return *domTree; }
		inline bool dominates(const llvm::BasicBlock * a, const llvm::BasicBlock * b) { return domTree->dominates(a, b); }

		inline llvm::PostDominatorTree & getPostDomTree() { return *postDomTree; }
		inline bool postDominates(const llvm::BasicBlock * a, const llvm::BasicBlock * b) { return postDomTree->dominates(a, b); }

		AnalysisStruct(AnalysisProvider & _provider, llvm::Function & _func, llvm::LoopInfo & _loopInfo, llvm::DominatorTree & _domTree, llvm::PostDominatorTree & _postDomTree) :
			provider(&_provider), func(&_func), loopInfo(&_loopInfo), domTree(&_domTree), postDomTree(&_postDomTree)
		{}

		void rebuild();
	};
}

#endif /* ANALYSISSTRUCT_HPP_ */
