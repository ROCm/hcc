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
 * CNSOperations.h
 *
 *  Created on: 29.04.2010
 *      Author: Simon Moll
 */

#ifndef CNSOPERATIONS_HPP_
#define CNSOPERATIONS_HPP_

#include <stack>

#include <axtor/util/llvmDuplication.h>

#include "BlockGraph.h"
#include "Bitmask.h"

namespace axtor {

namespace cns
{
	struct SED
	{
		uint iDom;
		BlockGraph::SubgraphMask mask;

		SED(uint _iDom, BlockGraph::SubgraphMask _mask) :
			iDom(_iDom), mask(_mask)
		{}
	};

	typedef std::vector<BlockGraph::SubgraphMask> MaskVector;
	typedef std::vector<SED> SEDVector;

	/*
	 * compute RC-nodes (this will destroy some edges in the graph)
	 */
	BlockGraph::SubgraphMask detectCandidateNodes(const BlockGraph::SubgraphMask & mask, const BlockGraph & graph);

	/*
	 * computes the SCCs of the graph (Strongly Connected Component)
	 */
	MaskVector computeSCCs(const BlockGraph::SubgraphMask & mask, const BlockGraph & graph);

	/*
	 * applies T1 and T2 to the maximum extend (CNS p. 47)
	 */
	void minimizeGraph(BlockGraph::SubgraphMask & mask, BlockGraph & graph);

	/*
	 * computes the dominance frontier
	 */
	BlockGraph::SubgraphMask computeDominanceRegion(const BlockGraph::SubgraphMask & mask, const BlockGraph & graph, uint node);
}

}

#endif /* CNSOPERATIONS_HPP_ */
