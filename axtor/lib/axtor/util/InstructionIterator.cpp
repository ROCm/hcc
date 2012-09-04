/*
 * InstructionIterator.cpp
 *
 *  Created on: 25.04.2010
 *      Author: gnarf
 */

#include <axtor/util/InstructionIterator.h>

namespace axtor {

InstructionIterator::InstructionIterator(llvm::Module & _mod) :
		mod(_mod)
	{
		func = mod.begin();
		block = func->begin();
		inst = block->begin();

		while (block == func->end() || inst == block->end()) operator++ ();
	}

	bool InstructionIterator::finished() const
	{
		return func == mod.end();
	}

	InstructionIterator InstructionIterator::operator++()
	{
		InstructionIterator old(*this);

		assert(func != mod.end() && "already reached end");

		if (inst == block->end()) {

			//try finding a new block (and init inst)
			do {
				++block;

				//finc a new function (and init block)
				while (block == func->end())
				{
					++func;
					if (finished())
						return old;

					block = func->begin();
				}

				inst = block->begin();
			} while (inst == block->end());

		} else {
			++inst;
		}

		return old;
	}

	llvm::Instruction * InstructionIterator::operator*()
	{
		return inst;
	}
}
