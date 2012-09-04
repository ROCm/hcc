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
 * TypeHandler.h
 *
 *  Created on: 19.03.2010
 *      Author: Simon Moll
 */

#ifndef TYPEHANDLER_HPP_
#define TYPEHANDLER_HPP_

#include <set>

#include <llvm/TypeSymbolTable.h>
#include <llvm/Instructions.h>


#include "AddressIterator.h"
#include "TypeStringBuilder.h"

#warning "deprecated"

namespace axtor {

struct TypeHandler
{
	/*
	 * return a name (debug)
	 */
	virtual std::string getName()=0;

	/*
	 * return a list of declarations that need to precede any function
	 */
	//virtual std::string getModuleEpilogue(llvm::TypeSymbolTable & symTable)=0;

	/*
	 * configure @builder so it evaluates to a name for this type
	 */
	virtual void getSymbol(TypeStringBuilder * builder, llvm::TypeSymbolTable & typeSymbols)=0;

	/*
	 * check whether this TypeHandler applies to to @type
	 */
	virtual bool appliesTo(const llvm::Type * type)=0;

	/*
 	 * register intrinsic descriptors with a platform
	 */
	virtual void registerWithPlatform(StringSet & nativeTypes, IntrinsicsMap & intrinsics, StringSet & derefFuncs)
	{}

	/*
	 * return a string built around @obj using values from @address
	 */
	virtual std::string dereference(std::string obj, const llvm::Type * objType, AddressIterator *& address, FunctionContext & funcContext, const llvm::Type* & oElementType)=0;
};

typedef std::set<TypeHandler*> TypeHandlerSet;

}

#endif /* TYPEHANDLER_HPP_ */
