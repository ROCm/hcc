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
 * GraphTypes.h
 *
 *  Created on: 19.02.2010
 *      Author: Simon Moll
 */

#ifndef GRAPHTYPES_HPP_
#define GRAPHTYPES_HPP_

#include <vector>
#include <map>

#include <llvm/BasicBlock.h>

/*
 * classes for a generic node-labeled directed graph
 * note that successors can be NULL
 */
template<class T>
class DirectedGraph
{
public:
	class GraphNode;
	typedef std::vector<GraphNode*> NodeVector;
	typedef std::set<GraphNode*> NodeSet;
	typedef std::set<DirectedGraph*> GraphSet;

	/*
	 * generic labeled graph node
	 */
	class GraphNode
	{
	private:
		uint index;
		T * label;
		NodeVector succ;

	protected:
		friend class DirectedGraph;
		void setIndex(uint _index) //to be exclusively called by the DirectedGraph constructor
		{
			index = _index;
		}

	public:
		uint getIndex()
		{
			return index;
		}

		NodeVector & getSuccessors() { return succ; }

		uint getNumSuccessors() { return succ.size(); }

		/*
		 * returns a successor
		 * Note, that in a graph edges can be disabled by setting a successor to NULL
		 */
		GraphNode * getSuccessor(int i)
		{
			return succ[i];
		}

		void addSuccessor(GraphNode * next)
		{
			succ.push_back(next);
		}

		void setSuccessor(uint id, GraphNode * val)
		{
			succ[id] = val;
		}

		GraphNode(T * _label) :
			label(_label)
		{}

		T * getLabel()
		{
			return label;
		}

		void print(std::ostream & out)
		{
			if (llvm::isa<llvm::BasicBlock>(label))
				out << label->getName().str();
			else
				out << label;
		}
	};

	/*
	 * utility class for simpler successor traversal, NULL successors are skipped on the fly
	 */
	class edge_iterator
	{
		GraphNode * parent;
		uint i;

	public:
		edge_iterator(const edge_iterator &src) :
			parent(src.parent),
			i(src.i)
		{}

		edge_iterator(GraphNode * _parent) :
			parent(_parent),
			i(0)
		{}

		void disableEdge()
		{
			parent->setSuccessor(i, NULL);
		}

		GraphNode * getTarget()
		{
			return parent->getSuccessor(i);
		}

		GraphNode * getBase()
		{
			return parent;
		}

		edge_iterator operator++()
		{
			edge_iterator copy(*this);

			//find next
			for(++i; i < parent->getNumSuccessors() && parent->getSuccessor(i) == NULL; ++i) {}

			return copy;
		}

		bool edgesLeft()
		{
			return i < parent->getNumSuccessors();
		}
	};

	typedef std::vector<edge_iterator> EdgeVector;

private:
	typedef std::map<T*, uint> NodeMap;
	NodeMap nodeMap;

	NodeVector nodes;
	int entryIdx;

public:
	DirectedGraph(NodeVector & _nodes, T * entryLabel) :
		nodes(_nodes),
		entryIdx(-1)
	{
		typename NodeVector::iterator it;
		uint nodeIdx = 0;
		for(
				nodeIdx = 0, it = nodes.begin();
				it != nodes.end();
				++it, ++nodeIdx)
		{
			T * label = (*it)->getLabel();

			//find the entry labeled node
			if (label == entryLabel) {
				entryIdx = nodeIdx;
			}

			//establish mapping from label to node
			nodeMap[label] = nodeIdx;

			//enumerate nodes
			(*it)->setIndex(nodeIdx);
		}
		assert((uint) entryIdx < nodes.size() && "invalid entry marker");
	}

	/*
	 * returns NULL if no entry was specified
	 */
	GraphNode * getEntry()
	{
		if (entryIdx >= 0)
			return nodes[entryIdx];
		else
			return NULL;
	}

	GraphNode * getNode(uint i)
	{
		return nodes[i];
	}

	uint size()
	{
		return nodes.size();
	}

	GraphNode * getNodeByLabel(T * label)
	{
		typename NodeMap::iterator it = nodeMap.find(label);

		if (it == nodeMap.end())
			return NULL;
		else
			return nodes[it->second];
	}

	EdgeVector getAllEdgesTo(GraphNode * target)
	{
		EdgeVector edges;

		for(typename NodeVector::iterator it = nodes.begin(); it != nodes.end(); ++it)
		{
			for(edge_iterator edge(*it); edge.edgesLeft(); edge++)
			{
				if (edge.getTarget() == target)
					edges.push_back(edge);
			}
		}

		return edges;
	}

	void dump()
	{
		std::cerr << "----- graph dump -----\n";
		uint i = 0;
		for (typename NodeVector::iterator it = nodes.begin(); it != nodes.end(); ++it, ++i)
		{
			T * label = (*it)->getLabel();
			if (llvm::isa<llvm::BasicBlock>(label))
			{
				std::cerr << '[' << i << "] : "; (*it)->print(std::cerr); std::cerr << " ->";
				
				for(edge_iterator edge(*it); edge.edgesLeft(); ++edge)
				{
					GraphNode * node = edge.getTarget();
					std::cerr << " ";
					node->print(std::cerr);;
				}
				std::cerr << "\n";
			}
		}
		std::cerr << "----------------------\n";
	}
};

#endif /* GRAPHTYPES_HPP_ */
