/*
 * ShortCircuitNodes.cpp
 *
 *  Created on: 29.11.2010
 *      Author: gnarf
 */

#include <axtor/ast/ShortCircuitNodes.h>

#ifdef ENABLE_SHORT_CIRCUIT_EXPRESSIONS

namespace axtor {

	ShortCircuitNode::CircuitBlock::CircuitBlock(llvm::BasicBlock * _block) :
			block(_block)
	{}

	/*
	 * circuit expression element for expressing short-circuit evaluated expressions
	 */
	ShortCircuitNode::CircuitExpression::CircuitExpression(CircuitElement * _startElement, CircuitElement * _primaryExit, CircuitElement * _defaultExit) :
			startElement(_startElement),
			primaryExit(_primaryExit),
			defaultExit(_defaultExit)
	{}


	ShortCircuitNode::ShortCircuitNode()
	{}

}

#endif /* ENABLE_SHORT_CIRCUIT_EXPRESSIONS */

