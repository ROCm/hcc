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
 * PlatformInfo.h
 *
 *  Created on: 27.02.2010
 *      Author: Simon Moll
 */

#ifndef PLATFORMINFO_HPP_
#define PLATFORMINFO_HPP_

#include <llvm/Function.h>

#include <axtor/intrinsics/IntrinsicDescriptors.h>
#include <axtor/util/stringutil.h>
#include <axtor/CommonTypes.h>
#include <axtor/console/CompilerLog.h>

//#include <axtor/intrinsics/TypeHandler.h>

namespace axtor {

/*
 * generic class that describes built-in functions and types
 */
class PlatformInfo
{
	//## Substitute types ##
	//maps llvm-types to platform specific type names
	StringSet nativeTypes;        /* OpaqueTypes implemented by this platform */
	//TypeHandlerSet handlers;      /* classes for type substitution */
	//TypeHandler * defaultHandler; /* default (fall back) type substitute */
	IntrinsicsMap intrinsics;     /* intrinsic functions */
	StringSet derefFuncs;         /* functions that act as dereferencing operators */

public:
	StringSet & getDerefFuncs();

	PlatformInfo(StringSet _nativeTypes, IntrinsicsMap _intrinsics);

public:
	bool lookUp(const llvm::Type * type, std::string typeName, std::string & out);

	/*
	 * calls the corresponding build method from the intrinsics map
	 */
	std::string build(std::string name, StringVector::const_iterator start, StringVector::const_iterator end);

	/*
	 * check if this type is intrinsic
	 */
	//bool implements(const llvm::Type * type);

	bool implements(const llvm::Type * type, std::string typeName);

	/*
	 * check if this function is intrinsic
	 */
	bool implements(llvm::GlobalValue * gv);

	/*
	 * return true if the associated intrinsic descriptor has class T
	 */
	template<class T>
	inline bool isIntrinsic(llvm::Function * func)
	{
		IntrinsicDescriptor * desc = intrinsics[func->getName().str()];
		return dynamic_cast<T*>(desc);
	}

	//TypeHandler * getDefaultHandler();
};

}

#endif /* PLATFORMINFO_HPP_ */
