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
 * Tarjan.h
 *
 *  Created on: 22.02.2010
 *      Author: Simon Moll
 */

#ifndef TARJAN_HPP_
#define TARJAN_HPP_

#include "GraphTypes.h"

/*
template<class T>
int getStackIndex(std::vector< typename DirectedGraph<T>::edge_iterator > & stack, typename DirectedGraph<T>::GraphNode * node)
{
	for(int i = 0; i < stack.size(); ++i)
	{
		if (stack[i].getTarget() == node)
			return i;
	}

	return -1;
}
*/

template<class T>
class Tarjan
{
	typedef DirectedGraph<T> Graph;
	typedef typename Graph::NodeSet* Color;

	Graph * graph;

	//### node color management ###
	Color * nodeColors;
	std::vector<Color> subSets;

	Color getColor(typename Graph::GraphNode * node)
	{
		return nodeColors[node->getIndex()];
	}

	Color createColor()
	{
		Color color = new typename Graph::NodeSet();
		subSets.push_back(color);
		std::cerr << "created color\n";
		return color;
	}

	uint getNextUncoloredNode(uint start)
	{
		uint i = start;

		for(; i < graph->size(); ++i)
		{
			if (! nodeColors[i])
				return i;
		}

		return i;
	}

	//### Tarjan stack ###

	struct SCCJob
	{
		SCCJob * parent;

		typename Graph::edge_iterator edge;
		Color explorationColor;

		SCCJob() :
			parent(NULL),
			edge(NULL),
			explorationColor(NULL)
		{}

		SCCJob(SCCJob * _parent, typename Graph::edge_iterator _edge, Color _explorationColor) :
			parent(_parent),
			edge(_edge),
			explorationColor(_explorationColor)
		{}

		typename Graph::GraphNode * getTarget()
		{
			return edge.getTarget();
		}

		uint getBaseIndex()
		{
			return edge.getBase()->getIndex();
		}

		//### Auxiliary methods for path coloring ###

		void paint(Color color)
		{
			typename Graph::GraphNode * node = edge.getBase();
			color->insert(node);
			explorationColor = color;
		}

		Color getBaseColor(Tarjan & tarjan)
		{
			return tarjan.getColor(edge.getBase());
		}

		/*
		 * changes this job to a job for a neighbouring edge
		 */
		bool flipToNextEdge(Tarjan & tarjan)
		{
			if (! edge.edgesLeft())
				return false;

			++edge;
			if (parent) {
				explorationColor = parent->explorationColor;

			} else {
				explorationColor = getBaseColor(tarjan);

			}

			return true;
		}

		/*
		 * paint a back until a specific node was reached
		 */
		void paintBackToNode(typename Graph::GraphNode * target, Color color)
		{
			SCCJob * curr = this;

			do
			{
				curr->paint(color);
				curr = curr->parent;
			} while(curr && curr->getTarget() != target);
		}

		/*
		 * paints everything back in color until a node is reached that already has this color
		 */
		void paintBackToColoredNode(Tarjan & tarjan, Color color)
		{
			SCCJob * curr = this;

			while (curr->getBaseColor(tarjan) != color)
			{
				curr->paint(color);
				curr = curr->parent;
				assert(curr && "must not reach NULL");
			};
		}
	};

	//Tarjan processing stack
	class JobStack
	{
		SCCJob * stack;
		std::vector<bool> stacked;
		uint top;

	public:
		JobStack(uint size) :
			stack(new SCCJob[size]),
			stacked(size, false),
			top(0u)
		{}

		~JobStack()
		{
			delete [] stack;
		}

		void push(SCCJob * parent, typename Graph::edge_iterator edge, Color explorationColor)
		{
			stacked[edge.getBase()->getIndex()] = true;
			stack[top++] = SCCJob(parent, edge, explorationColor);
		}

		SCCJob * back()
		{
			return stack + top - 1;
		}

		void pop()
		{
			SCCJob * job = &( stack[--top] );
			uint index = job->getBaseIndex();
			stacked[index] = false;
		}

		bool hasElements() { return top > 0; }

		bool contains(typename Graph::GraphNode * node)
		{
			uint index = node->getIndex();
			return stacked[index];
		}
	};



public:
	Tarjan(Graph & _graph) :
		graph(&_graph)
	{
		nodeColors = new Color[graph->size()];
		memset(nodeColors, 0, graph->size());
	}

	~Tarjan()
	{
		delete [] nodeColors;
	}

	//### core method ###
	void generateSCCs()
	{
		//stack
		JobStack stack(graph->size());         //stack of edges
		uint entry = 0;

		stack.push(NULL, typename Graph::edge_iterator( graph->getNode(entry) ), NULL);
		std::vector<bool> visited(graph->size(), false);

		do
		{
			while (stack.hasElements())
			{
				//# fetch
				SCCJob * curr = stack.back();

				visited[curr->getBaseIndex()] = true;

				//# process
				typename Graph::GraphNode * target = curr->getTarget();
				Color targetColor = getColor(target);

				//# found a new cycle
				if (stack.contains(target) && targetColor == NULL && curr->explorationColor == NULL)
				{
					Color color = createColor();
					curr->paintBackToNode(target, color);

				//# dipping streak
				} else if (targetColor == curr->explorationColor) {
					curr->paintBackToColoredNode(*this, curr->explorationColor);

				//# explore all pathes
				} else {
					if (curr->flipToNextEdge(*this)) {
						target = curr->getTarget();
						if (! visited[target->getIndex()]) {
							stack.push(curr, typename Graph::edge_iterator(target), curr->explorationColor);
						}

					} else {
						stack.pop();
					}
				}
			}

			//find next uninspected node
			for(;entry < graph->size() && visited[entry]; ++entry) {}

		} while (entry < graph->size());
	}
};
#endif /* TARJAN_HPP_ */
