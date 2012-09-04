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
 * Regularizer.h
 *
 *  Created on: 29.04.2010
 *      Author: Simon Moll
 */

#ifndef REGULARIZER_HPP_
#define REGULARIZER_HPP_

#include <axtor/config.h>

#include <llvm/Pass.h>

#include <axtor/util/BlockCopyTracker.h>
#include <axtor/pass/TargetProvider.h>

#include <axtor/pass/OpaqueTypeRenamer.h>
#include <axtor/cns/BlockGraph.h>
#include <axtor/cns/CNS.h>
#include <axtor/cns/CNSScoring.h>
#include <axtor/cns/SplitTree.h>
#include <axtor/util/ResourceGuard.h>
#include <axtor/util/llvmDuplication.h>



/*
 * The Regularizer makes irreducible control flow reducible by applying controlled node splitting
 */
namespace axtor {

class CNSPass : public llvm::ModulePass
{
	cns::SplitTree * generateSplitSequence(cns::SplitTree * root, BlockGraph::SubgraphMask & mask, BlockGraph & graph);

	void applySplitSequence(BlockCopyTracker & tracker, BlockGraph & graph, std::vector<uint> nodes) const;

	bool runOnFunction(BlockCopyTracker & tracker, llvm::Function & func);
public:
	static char ID;

	CNSPass();

	virtual void getAnalysisUsage(llvm::AnalysisUsage & usage) const;

	bool runOnModule(llvm::Module & M);

	virtual const char * getPassName() const;
};

}


#endif /* REGULARIZER_HPP_ */
