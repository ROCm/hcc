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
 * AddressIterator.h
 *
 *  Created on: 19.03.2010
 *      Author: Simon Moll
 */

#ifndef ADDRESSITERATOR_HPP_
#define ADDRESSITERATOR_HPP_

#include <llvm/Value.h>
#include <axtor/util/llvmConstant.h>
#include <axtor/CommonTypes.h>

namespace axtor {

/*
 * Auxiliary class
 */
class AddressIterator
{
	llvm::Value * val;
	AddressIterator * next;
	bool cumulativeValue;

	AddressIterator();
	AddressIterator(AddressIterator * _next, llvm::Value * _val, bool _isCumulative);


public:
	void dump(int idx);
	void dump();

	llvm::Value * getValue();

	AddressIterator * getNext();

	bool isEnd() const;

	bool isEmpty() const;

	// true, if this is an addition operating on a GEP (without dereferencing the type)
	// Note, that added zeros in cascading GEPs will be eliminated to begin with
	bool isCumulative() const;

	/*
	 * returns a pair consisting of the dereferenced root object and a corresponding index value list
	 */

	struct AddressResult
	{
		llvm::Value * rootValue;
		AddressIterator * iterator;

		AddressResult(llvm::Value * _rootValue, AddressIterator * _iterator);
	};

	static AddressResult createAddress(llvm::Value * derefValue, StringSet & derefFuncs);
};

}

#endif /* ADDRESSITERATOR_HPP_ */
