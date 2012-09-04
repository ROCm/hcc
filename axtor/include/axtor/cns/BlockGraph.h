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
 * BlockGraph.h
 *
 *  Created on: 26.04.2010
 *      Author: Simon Moll
 */

#ifndef BLOCKGRAPH_HPP_
#define BLOCKGRAPH_HPP_

#include <axtor/config.h>

#include <vector>
#include <map>
#include <iostream>

#include <llvm/Function.h>
#include <llvm/BasicBlock.h>
#include <llvm/Support/CFG.h>

#include <axtor/util/stringutil.h>
#include "Bitmask.h"

namespace axtor {

class BlockGraph
{
public:
	typedef BoolVector SubgraphMask;

private:
	typedef std::vector<BoolVector> DirectedGraph;
	typedef std::vector<llvm::BasicBlock*> BlockVector;

	DirectedGraph graph;
	BlockVector labels;

	BlockGraph(DirectedGraph _graph, BlockVector _labels);

	void resetGraph(uint size);

	BlockGraph(uint size){
		resetGraph(size);
	}

	DirectedGraph createExtendedGraph(SubgraphMask & mask, uint size) const;
	static void copyLeavingEdges(DirectedGraph & destGraph, uint destNode, DirectedGraph sourceGraph, uint sourceNode);
	static void copyIncomingEdges(DirectedGraph & destGraph, uint destNode, DirectedGraph sourceGraph, uint sourceNode);

public:

	inline void setLabel(uint index, llvm::BasicBlock * _label)
	{
		assert(index < labels.size());
		labels[index] = _label;
	}

	inline llvm::BasicBlock * getLabel(uint index) const
	{
		assert(index < labels.size());
		return labels[index];
	}

	inline BlockGraph::SubgraphMask mergedPredecessors(SubgraphMask mask) const
	{
		SubgraphMask result(getSize(), false);

		for(uint i = 0 ; i < getSize(); ++i)
		{
			if (mask[i])
			{
				result = OR(result, graph[i]);
			}
		}

		return result;
	}

	inline BlockGraph::SubgraphMask mergedSuccessors(SubgraphMask mask) const
	{
		SubgraphMask result(getSize(), false);

		for(uint i = 0 ; i < getSize(); ++i)
		{
			if (!mask[i] && EXISTS(AND(graph[i], mask)))
			{
				result[i] = true;
			}
		}

		return result;
	}
	inline bool getEdge(const SubgraphMask & mask, uint from, uint to) const
	{
		return mask[to] && mask[from] && graph[to][from];
	}

	inline void setEdge(uint from, uint to, bool value)
	{
		graph[to][from] = value;
	}

	inline void setAllEdges(const SubgraphMask & sources, const SubgraphMask & targets, bool value)
	{
		for(uint in = 0; in < sources.size(); ++in)
		if (sources[in])
			for(uint out = 0; out < targets.size(); ++out)
			if (targets[out])
				setEdge(in, out, value);
	}

	inline void setAllEdges(uint source, const SubgraphMask & targets, bool value)
	{
		for(uint out = 0; out < targets.size(); ++out)
		if (targets[out])
			setEdge(source, out, value);
	}

	inline void setAllEdges(const SubgraphMask & sources, uint target, bool value)
	{
		for(uint in = 0; in < sources.size(); ++in)
		if (sources[in])
			setEdge(in, target, value);
	}


	llvm::BasicBlock * getBasicBlock(uint index);
	std::string getNodeName(uint index) const;
	SubgraphMask createMask() const;

	// BlockGraph(llvm::Function & func);

	static BlockGraph CreateFromFunction(llvm::Function & func, SubgraphMask & oMask);

	~BlockGraph();

	/*
	 * splits the node at @index in the graph. @mask will be adapted to the returned graph
	 */
	BlockGraph createSplitGraph(SubgraphMask & mask, uint index);

	uint getSize(const SubgraphMask & mask) const;
	uint getSize() const;

	inline SubgraphMask getPredecessors(SubgraphMask mask, uint node) const
	{
		return AND(mask, graph[node]);
	}

	uint getNumPredecessors(const SubgraphMask & mask, uint index) const;
	uint getNumPredecessorsNonReflexive(const SubgraphMask & mask, uint index) const;

	uint getNumSuccessors(const SubgraphMask & mask, uint index) const;
	//uint getNumSuccessorsNonReflexive(const SubgraphMask & mask, uint index) const;
	/*
	 * returns -1 if there is not a unique incoming edge
	 */
	int getSinglePredecessorNonReflexive(const SubgraphMask & mask, uint index) const;
	int getSingleSuccessorNonReflexive(const SubgraphMask & mask, uint index) const;


	//transformations
	void removeNode(SubgraphMask & mask, uint index) const;
	void contractNode(SubgraphMask & mask, uint mergedNode, uint destNode);

	/*
	 * removes all edges from the E that contain vertices not occurring in the subgraph mask
	 */
	void enforceSubgraph(SubgraphMask & mask);

	void dumpGraphviz(const SubgraphMask & mask, std::ostream & out) const;
	void dump(const SubgraphMask & mask) const;
	static void dumpInternal(const DirectedGraph & graph);

	SubgraphMask getExitingNodes(const SubgraphMask & scope, const SubgraphMask & mask) const;
	SubgraphMask getEntryNodes(const SubgraphMask & scope, const SubgraphMask & mask) const;

	BlockGraph transposed() const;
};

}

#endif /* BLOCKGRAPH_HPP_ */
