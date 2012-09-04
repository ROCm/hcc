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
 * InstructionIterator.h
 *
 *  Created on: 13.04.2010
 *      Author: Simon Moll
 */

#ifndef INSTRUCTIONITERATOR_HPP_
#define INSTRUCTIONITERATOR_HPP_

#include <llvm/Module.h>

namespace axtor {

class InstructionIterator
{
	llvm::Module & mod;

	llvm::Module::iterator func;
	llvm::Function::iterator block;
	llvm::BasicBlock::iterator inst;

public:
	InstructionIterator(llvm::Module & _mod);

	bool finished() const;

	InstructionIterator operator++();

	llvm::Instruction * operator*();
};

}

#endif /* INSTRUCTIONITERATOR_HPP_ */
