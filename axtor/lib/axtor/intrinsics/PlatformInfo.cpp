/*
 * PlatformInfo.cpp
 *
 *  Created on: 25.04.2010
 *      Author: gnarf
 */

#include <axtor/intrinsics/PlatformInfo.h>

namespace axtor {



StringSet & PlatformInfo::getDerefFuncs()
{
	return derefFuncs;
}

PlatformInfo::PlatformInfo(StringSet _nativeTypes, IntrinsicsMap _intrinsics) :
	nativeTypes(_nativeTypes),
	intrinsics(_intrinsics)
{
	/*for (TypeHandlerSet::iterator itHandler = handlers.begin(); itHandler != handlers.end(); ++itHandler)
	{
		TypeHandler * handler = *itHandler;
		handler->registerWithPlatform(nativeTypes, intrinsics, derefFuncs);
	}

	if (defaultHandler)
	{
		defaultHandler->registerWithPlatform(nativeTypes, intrinsics, derefFuncs);
	}*/
}

bool PlatformInfo::lookUp(const llvm::Type * type, std::string typeName, std::string & out)
{
#ifdef DEBUG
	std::cerr << "PlatformInfo::lookUp : " << typeName << "\n";
#endif
	/*if (llvm::isa<const llvm::OpaqueType>(type)) {
		if (nativeTypes.find(typeName) != nativeTypes.end()) {
			out = typeName;
			return true;
		}
	}*/

	return false;
}


/*
 * calls the corresponding build method from the intrinsics map
 */
std::string PlatformInfo::build(std::string name, StringVector::const_iterator start, StringVector::const_iterator end)
{
	IntrinsicDescriptor * desc = intrinsics[name];
	assert(desc && "unsupported intrinsic");

	return desc->build(start, end);
}

/*
 * check if this type is intrinsic
 */
/*bool PlatformInfo::implements(const llvm::Type * type)
{
	for(TypeHandlerSet::iterator it = handlers.begin(); it != handlers.end(); ++it)
	{
		if ((*it)->appliesTo(type))
			return true;
	}
	return false;
}*/

bool PlatformInfo::implements(const llvm::Type * type, std::string typeName)
{
	//return (llvm::isa<llvm::OpaqueType>(type) && nativeTypes.find(typeName) != nativeTypes.end());/* ||
			//implements(type);*/
  return false;
}

/*
 * check if this function is intrinsic
 */
bool PlatformInfo::implements(llvm::GlobalValue * gv)
{
	return intrinsics.find(gv->getName()) != intrinsics.end();
}

/*
 * return the unique SubstituteType object for this type (if none applies, use @defaultSubstitute instead)
 */
/*TypeHandler * PlatformInfo::lookUpHandler(const llvm::Type * type)
{
	TypeHandler * handler = NULL;

	//seek a unique SubstituteType
	for(TypeHandlerSet::iterator it = handlers.begin(); it != handlers.end(); ++it)
	{
		TypeHandler * candidate = *it;
		if (candidate->appliesTo(type))
		{
			//std::cerr << "matching candidate " << candidate->getName() << "\n";

 			if (candidate && handler)
				Log::fail(type, "multiple type handlers could be applied." + candidate->getName() + " OR " + handler->getName());

			handler = candidate;
		}
	}

	return handler;
}

TypeHandler * PlatformInfo::getDefaultHandler()
{
	return defaultHandler;
}*/

}
