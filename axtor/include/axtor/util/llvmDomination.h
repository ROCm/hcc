
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
 * llvmDomination.h
 *
 *  Created on: 06.03.2010
 *      Author: Simon Moll
 */

#ifndef LLVMDOMINATION_HPP_
#define LLVMDOMINATION_HPP_

#include <llvm/Analysis/Dominators.h>

#include <axtor/CommonTypes.h>

namespace axtor {

typedef std::vector<llvm::DomTreeNode*> DomTreeNodeVector;

/*
 * computes the intersection of all Dominance Frontiers of all elements in @nodes
 */
/*template<typename IT>
BlockSet computeCommonDomFront(llvm::DominanceFrontiert & domFront, IT begin, IT end)
{
	BlockSet result;
	bool wasMerged = false;

	for(IT itNode = begin; itNode != end; ++itNode)
	{
		llvm::BasicBlock * node = *itNode;
		BlockSet nodeDomFront = domFront[node];

		if (wasMerged) {
			BlockSet tmp;
			std::set_intersection(result.begin(), result.end(), nodeDomFront.begin(), nodeDomFront.end(), tmp.begin());
			result.swap(tmp);
		} else {
			result.swap(nodeDomFront);
		}
	}

	return result;
} */

/*
 * collect all blocks that are reachable and post-dominated from @entryBlock
 */
BlockSet getSelfDominatedBlocks(llvm::BasicBlock * entryBlock, llvm::PostDominatorTree & postDomTree);

bool dominatesAll(llvm::DominatorTree & domTree, llvm::DomTreeNode * node, const BlockSet & blocks);

llvm::DomTreeNode * findImmediateDominator(llvm::DominatorTree & domTree, const BlockSet & blocks);

BlockSet computeDominatedRegion(llvm::DominatorTree & domTree, llvm::BasicBlock * header, BlockSet exits);

}

#endif /* LLVMDOMINATION_HPP_ */
