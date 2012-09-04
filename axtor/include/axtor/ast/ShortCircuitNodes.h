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
 * ShortCircuitNodes.h
 *
 *  Created on: 29.11.2010
 *      Author: Simon Moll
 */

#ifndef SHORTCIRCUITNODES_HPP_
#define SHORTCIRCUITNODES_HPP_

#include <axtor/config.h>

#ifdef ENABLE_SHORT_CIRCUIT_EXPRESSIONS
#include <vector>
#include <axtor/ast/ASTNode.h>


namespace axtor {

namespace ast {

/*
 * control-flow node for representing a short-circuit boolean expression
 */
struct ShortCircuitNode : public ConditionalNode
{
	struct CircuitElement
	{
		virtual bool isBlockWrapper() const;
	};

	/*
	 * wrapper for a single basic block
	 */
	struct CircuitBlock : public CircuitElement
	{
		CircuitBlock(llvm::BasicBlock * _block);
		bool isBlockWrapper() const
		{
			return true;
		}

		llvm::BasicBlock * getBlock() const { return block; }

	private:
		llvm::BasicBlock * block;
	};

	/*
	 * circuit expression element for expressing short-circuit evaluated expressions
	 */
	struct CircuitExpression : public CircuitElement
	{
		CircuitExpression(CircuitElement * _startElement, CircuitElement * _primaryExit, CircuitElement * _defaultExit);
		bool isBlockWrapper() const
		{
			return false;
		}

		CircuitElement * getStartElement() const { return startElement; }
		CircuitElement * getPrimaryExit() const { return primaryExit; }
		CircuitElement * getDefaultExit() const { return defaultExit; }

	private:
		CircuitElement * startElement;
		CircuitElement * primaryExit;
		CircuitElement * defaultExit;
	};

private:
	std::vector<CircuitElement*> elements;
};

}

}

#endif

#endif  /* SHORTCIRCUITNODES_HPP_ */
