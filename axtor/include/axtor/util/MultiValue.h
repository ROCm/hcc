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
 * MultiValue.h
 *
 *  Created on: 16.03.2010
 *      Author: Simon Moll
 */

#ifndef MULTIVALUE_HPP_
#define MULTIVALUE_HPP_

#include <algorithm>

#include <axtor/CommonTypes.h>

namespace axtor {

class MultiValue
{
public:
	typedef void (*ValueOperation)(llvm::Value * val);

private:
	class ValueFunctor : public std::unary_function<llvm::Value*,void>
	{
		ValueOperation op;
	public:
		ValueFunctor(ValueOperation _op);

		void operator()(llvm::Value * val);
	};

// ### FUNCTORS ###
	static void func_dropReferences(llvm::Value * val);

	static void func_erase(llvm::Value * val);

	static void func_removeConstantUsers(llvm::Value * val);

	static void func_dump(llvm::Value * val);

public:
	static void apply(ValueSet & values, ValueOperation op);

	static void erase(ValueSet & values);

	static void dump(ValueSet & values);
};

}


#endif /* MULTIVALUE_HPP_ */
