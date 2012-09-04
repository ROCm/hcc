/*
 * CNSScoring.cpp
 *
 *  Created on: 29.04.2010
 *      Author: gnarf
 */

#include <axtor/cns/CNSScoring.h>


namespace axtor
{

namespace cns
{

typedef uint (*ScoringFunction)(BlockGraph::SubgraphMask & mask, BlockGraph & graph, uint candidate);

uint scoreNumInstructions(BlockGraph::SubgraphMask & mask, BlockGraph & graph, uint candidate)
{
	llvm::BasicBlock * block = graph.getBasicBlock(candidate);
	return block->getInstList().size();
}

uint scoreBranches(BlockGraph::SubgraphMask & mask, BlockGraph & graph, uint candidate)
{
	return graph.getNumSuccessors(mask, candidate);
}


uint getLowestScoringNode(BlockGraph::SubgraphMask & mask, BlockGraph & graph, ScoringFunction heuristicFunc)
{
	uint lowest = 0xFFFFFFFF;
	uint lowestNode = 0;

	for(uint i = 0; i < mask.size(); ++i)
	{
		if (mask[i]) {
			uint tmpScore = (*heuristicFunc)(mask, graph, i);
			if (tmpScore < lowest) {
				lowest = tmpScore;
				lowestNode = i;
			}
		}
	}

	return lowestNode;
}

}

}
