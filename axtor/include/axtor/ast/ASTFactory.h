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
 * ASTFactory.h
 *
 *  Created on: 12.06.2010
 *      Author: Simon Moll
 */

#ifndef ASTFACTORY_HPP_
#define ASTFACTORY_HPP_

#include "BasicNodes.h"

namespace axtor {
namespace ast {

struct ASTFactory
{
	static FunctionNode * createFunction(llvm::Function * _func, ControlNode * body);
	static ControlNode * createConditional(llvm::BasicBlock * block, ControlNode * _onTrue, ControlNode * _onFalse);

	static ControlNode * createBlock(llvm::BasicBlock * block);
	static ControlNode * createList(const NodeVector & nodes);

	static ControlNode * createInfiniteLoop(ControlNode * child);
	static ControlNode * createBreak(llvm::BasicBlock * block);
	static ControlNode * createBreak();
	static ControlNode * createContinue(llvm::BasicBlock * block);
	static ControlNode * createContinue();

	static ControlNode * createReturn(llvm::BasicBlock * block);
	static ControlNode * createUnreachable(llvm::BasicBlock * block);
};

}
}

#endif /* ASTFACTORY_HPP_ */
