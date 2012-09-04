/*
 * BlockGraph.cpp
 *
 *  Created on: 26.04.2010
 *      Author: gnarf
 */


#include <axtor/cns/BlockGraph.h>
#include <assert.h>
#include <llvm/Support/raw_ostream.h>



namespace axtor {

BlockGraph::BlockGraph(DirectedGraph _graph, BlockVector _labels) :
		graph(_graph), labels(_labels)
{
#if 0
//#ifdef DEBUG
	uint expectedSize = graph.size();
	for(DirectedGraph::iterator it = graph.begin(); it != graph.end(); ++it)
	{
		assert(expectedSize == it->size()); //"edge vector size differs from number of nodes"
	}
#endif
	assert(graph.size() == labels.size());// && "not all nodes are labeled"
}

BlockGraph::~BlockGraph()
{}


BlockGraph BlockGraph::CreateFromFunction(llvm::Function & func, BlockGraph::SubgraphMask & oMask)
{
	typedef llvm::GraphTraits<llvm::BasicBlock*> CFG;
	typedef std::map<llvm::BasicBlock*,int> IndexMap;

	uint size = func.getBasicBlockList().size();
	BlockGraph graph(size);

	graph.resetGraph(size);
	oMask = SubgraphMask(size, false);

	IndexMap indexMap;

	//build index map
#ifdef DEBUG
	llvm::errs() << "BlockGraph::BlockGraph(LLVMFunction) {\n";
#endif
	{
		int index = 0;
		for(llvm::Function::iterator bb = func.begin(); bb != func.end(); ++bb, ++index)
		{
			llvm::BasicBlock * block = llvm::cast<llvm::BasicBlock>(bb);

			graph.labels[index] = block;
			indexMap[block] = index;
#ifdef DEBUG
			llvm::errs() << block->getName() << " @ " << index << "\n";
#endif
		}
	}
#ifdef DEBUG
	llvm::errs() << "} // BlockGraph::BlockGraph(LLVMFunction)\n";
#endif

	//add adjacency information
	{
		int startIndex = 0;
		for(llvm::Function::iterator bb = func.begin(); bb != func.end(); ++bb, ++startIndex)
		{
			llvm::BasicBlock * block = llvm::cast<llvm::BasicBlock>(bb);

			CFG::ChildIteratorType childBegin = CFG::child_begin(block);
			CFG::ChildIteratorType childEnd = CFG::child_end(block);
			bb->getTerminator()->dump();

			for(CFG::ChildIteratorType succ = childBegin; succ != childEnd; ++succ)	{
				llvm::BasicBlock * dest = *succ;

				int destIndex = indexMap[dest];

				oMask[startIndex] = true;
				oMask[destIndex] = true;
				graph.setEdge(startIndex, destIndex, true);
			}
		}
	}
#ifdef DEBUG
	graph.dumpGraphviz(oMask, std::cerr);
#endif

	return graph;
}

void BlockGraph::resetGraph(uint size)
{
	graph.resize(size, std::vector<bool>(size, false));
	labels.resize(size, NULL);
}

uint BlockGraph::getSize(const SubgraphMask & mask) const
{
	uint count = 0;
	for(uint i = 0; i < mask.size(); ++i)
		if (mask[i])
			++count;

	return count;
}

uint BlockGraph::getSize() const
{
	return graph.size();
}

BlockGraph::DirectedGraph BlockGraph::createExtendedGraph(SubgraphMask & mask, uint size) const
{
	if (graph.size() == size)
		return graph;

	assert((size > graph.size()) && "operation would crop data from the graph");

	DirectedGraph result;

	for(DirectedGraph::const_iterator it = graph.begin(); it != graph.end(); ++it)
	{
		BoolVector copy = *it;
		copy.resize(size, false);
		result.push_back(copy);
	}

	std::vector<bool> row(size, false);
	for(uint i = graph.size(); i < size; ++i)
	{
		result.push_back(row);
	}

	mask.resize(size, false);

	return result;
}

BlockGraph::SubgraphMask BlockGraph::createMask() const
{
	return SubgraphMask(getSize(), true);
}

uint BlockGraph::getNumSuccessors(const SubgraphMask & mask, uint index) const
{
	uint count = 0;
	for(uint i = 0; i < getSize(); ++i)
		if (getEdge(mask, index, i))
			++count;

	return count;
}

uint BlockGraph::getNumPredecessors(const SubgraphMask & mask, uint index) const
{
	uint count = 0;

	for(uint pred = 0; pred < getSize(); ++pred)
	{
		if (getEdge(mask, pred, index))
			++count;
	}
	return count;
}


uint BlockGraph::getNumPredecessorsNonReflexive(const SubgraphMask & mask, uint index) const
{
	uint count = 0;

	for(uint pred = 0; pred < getSize(); ++pred)
	{
		if (pred != index && getEdge(mask, pred, index))
			++count;
	}
	return count;
}

int BlockGraph::getSinglePredecessorNonReflexive(const SubgraphMask & mask, uint index) const
{
	int res = -1;

	for(uint i = 0; i < getSize(); ++i)
	{
		if (i != index && getEdge(mask, i, index)) {
			if (res > -1) {
				return -1;
			} else {
				res = (int) i;
			}
		}
	}

	return res;
}

int BlockGraph::getSingleSuccessorNonReflexive(const SubgraphMask & mask, uint index) const
{
	int res = -1;

	for(uint i = 0; i < getSize(); ++i)
	{
		if (i != index && getEdge(mask, index, i)) {
			if (res > -1) {
				return -1;
			} else {
				res = (int) i;
			}
		}
	}

	return res;
}

void BlockGraph::contractNode(SubgraphMask & mask, uint mergedNode, uint consumedNode)
{
	for(uint index = 0; index < mask.size(); ++index)
	{
		if (getEdge(mask, consumedNode, index))
			setEdge(mergedNode, index, true);
	}

	mask[consumedNode] = false;
}

void BlockGraph::removeNode(SubgraphMask & mask, uint index) const
{
	mask[index] = false;
}

/*std::_Bit_reference BlockGraph::operator()(int from, int to)
{
	return graph[to][from];
}*/

/*
 * copies over all edges that leave @sourceNode in @sourceGraph to @destNode in @destGraph
 *
 * O(n)
 */
void BlockGraph::copyLeavingEdges(DirectedGraph & destGraph, uint destNode, DirectedGraph sourceGraph, uint sourceNode)
{
#ifdef DEBUG
	std::cerr << "copyLeavingEdges(dest=" << destNode << ",source=" << sourceNode << ")\n";
#endif
	for(uint edge = 0; edge < sourceGraph.size(); ++edge)
	{
		if (sourceGraph[edge][sourceNode])
			destGraph[edge][destNode] = sourceGraph[edge][sourceNode];
	}
}
/*
* O(1)
*/
void BlockGraph::copyIncomingEdges(DirectedGraph & destGraph, uint destNode, DirectedGraph sourceGraph, uint sourceNode)
{
	assert((sourceGraph.size() == destGraph.size()) && "graphs must have equal size");
	destGraph[destNode] = sourceGraph[sourceNode];
}

/*
 * return a correctly labeled graph after splitting the node at @index
 */
BlockGraph BlockGraph::createSplitGraph(SubgraphMask & mask, uint index)
{
	BoolVector & incoming = graph[index];
	llvm::BasicBlock * nodeLabel = labels[index];

	uint numIncoming = getNumPredecessors(mask, index);

	if (numIncoming <= 1) {
#ifdef DEBUG
		std::cerr << "(!!!) attempted to split node with " << numIncoming << " predecessors" << std::endl;
#endif
		return *this;
	}



	uint graphSize = getSize() + numIncoming - 1;
	uint nextFreeIndex = getSize();

#ifdef DEBUG
	std::cerr << "---- SPLIT NODE ----- (destSize=" << graphSize << ",splitNode=" << index << ")\n";
	std::cerr << "mask : "; dumpVector(mask); std::cerr << "\n";
	dumpInternal(graph);
#endif

	DirectedGraph splitGraph = createExtendedGraph(mask, graphSize);
	BlockVector splitLabels = labels;
	splitLabels.resize(graphSize, nodeLabel);

#ifdef DEBUG
	std::cerr << "extended graph : "; dumpVector(mask); std::cerr << "\n{\n";
	dumpInternal(splitGraph);
	std::cerr << "}\n";
#endif

	{
		BoolVector::iterator it = incoming.begin();

		//fast forward to the first bit set (and skip it)
		uint start = 0;
		for(;(it != incoming.end()) && !*it; ++it, ++start) {}
		++it; ++start;

		assert((it != incoming.end()) && "checked before that there is more than one predecessor");

		//actual splitting
		for(; it != incoming.end(); ++it, ++start)
		{
			if (*it && mask[start]) {
#ifdef DEBUG
				std::cerr << "Splitting for incoming node=" << start << "\n";
#endif
				mask[nextFreeIndex] = true;

				//reattaching incoming edge
				splitGraph[index][start] = false; //remove the incoming edge to the splitted node
				splitGraph[nextFreeIndex][start] = true; //and instead attach it to the newly created offspring
#ifdef DEBUG
				std::cerr << "reattached entry to graph :"; dumpVector(mask); std::cerr << "\n{\n";
				dumpInternal(splitGraph);
#endif
				//copy over leaving edges
				copyLeavingEdges(splitGraph, nextFreeIndex, graph, index); //copy over all leaving edges
#ifdef DEBUG
				std::cerr << "after copying all leaving edges :"; dumpVector(mask); std::cerr << "\n{\n";
				dumpInternal(splitGraph);
#endif
				++nextFreeIndex;
			}
		}
	}

#ifdef DEBUG
	std::cerr << "resulting graph : "; dumpVector(mask); std::cerr << "\n{\n";
	dumpInternal(splitGraph);
	std::cerr << "}\n";
#endif

	return BlockGraph(splitGraph, splitLabels);
}

void BlockGraph::enforceSubgraph(SubgraphMask & mask)
{
	for(uint from = 0; from < getSize(); ++from)
		for(uint to = 0; to < getSize(); ++to)
		{
			if (! mask[from] || !mask[to])
			{
				setEdge(from, to, false);
			}
		}
}

std::string BlockGraph::getNodeName(uint index) const
{
	return labels[index]->getName().str();
}

void BlockGraph::dumpGraphviz(const SubgraphMask & mask, std::ostream & out) const
{
	out << "digraph BlockGraph {\n";
	for(uint from = 0; from < getSize(); ++from)
		for(uint to = 0; to < getSize(); ++to)
		{
			if (getEdge(mask, from, to)) {
				out << str<uint>(from) << "->" << str<uint>(to) << "\n";
			}
		}

	out << "}\n";
}

void BlockGraph::dump(const SubgraphMask & mask) const
{
	std::cerr << "BlockGraph {\n";
	for(uint from = 0; from < getSize(); ++from)
			for(uint to = 0; to < getSize(); ++to)
			{
				if (getEdge(mask, from, to)) {
					std::cerr << getNodeName(from) << "->" << getNodeName(to) << "\n";
				}
			}

		std::cerr << "}\n";
}

void BlockGraph::dumpInternal(const DirectedGraph & graph)
{
	for(uint from = 0; from < graph.size(); ++from) {
		for(uint to = 0; to < graph.size(); ++to) {
			if (graph[to][from])
				std::cerr << "1";
			else
				std::cerr << "0";
		}
		std::cerr << "\n";
	}
}

llvm::BasicBlock * BlockGraph::getBasicBlock(uint index)
{
	return labels[index];
}

BlockGraph::SubgraphMask BlockGraph::getExitingNodes(const SubgraphMask & scope, const SubgraphMask & mask) const
{
	SubgraphMask external = AND(scope, NOT(mask));
	return AND(scope, mergedPredecessors(external));

}

BlockGraph::SubgraphMask BlockGraph::getEntryNodes(const SubgraphMask & scope, const SubgraphMask & mask) const
{
	SubgraphMask external = AND(scope, NOT(mask));
	return AND(scope, mergedSuccessors(external));
}

BlockGraph BlockGraph::transposed() const
{
	DirectedGraph trans(getSize(), BoolVector(getSize(), false));

	for(uint i = 0; i < getSize(); ++i)
	{
		for(uint j = 0; j < getSize(); ++j)
		{
			trans[i][j] = graph[j][i];
		}
	}

	return BlockGraph(trans, labels);
}

}
