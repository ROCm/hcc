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
 * CNSScoring.h
 *
 *  Created on: 29.04.2010
 *      Author: Simon Moll
 */

#ifndef CNSSCORING_HPP_
#define CNSSCORING_HPP_

#include <axtor/cns/BlockGraph.h>

namespace axtor
{
	namespace cns
	{
		typedef uint (*ScoringFunction)(BlockGraph::SubgraphMask & mask, BlockGraph & graph, uint candidate);

		/*
		 * returns the amount of instructions in this basic block
		 */
		uint scoreNumInstructions(BlockGraph::SubgraphMask & mask, BlockGraph & graph, uint candidate);


		/*
		 * returns the number of predecessors of this block (in the BlockGraph)
		 */
		uint scoreBranches(BlockGraph::SubgraphMask & mask, BlockGraph & graph, uint candidate);


		/*
		 * takes a per-node based scoring function and returns the node that scores lowest
		 *
		 * returns 0 if the graph does not contain any nodes
		 */
		uint getLowestScoringNode(BlockGraph::SubgraphMask & mask, BlockGraph & graph, ScoringFunction heuristicFunc);

	}

}

#endif /* CNSSCORING_HPP_ */
