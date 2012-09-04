/*
 * ModuleInfo.cpp
 *
 *  Created on: 27.06.2010
 *      Author: gnarf
 */

#include <axtor/metainfo/ModuleInfo.h>
#include <axtor/console/CompilerLog.h>

namespace axtor
{
	ModuleInfo::ModuleInfo(llvm::Module & _M)  :
		BlockCopyTracker(_M),
		M(_M)
	{
		std::string errInfo;
		M.MaterializeAll(&errInfo);
		if (errInfo != "") {
			Log::fail("llvm::MaterializeAll error: " + errInfo);
		}
	}

	const llvm::Module * ModuleInfo::getModule() const
	{
		return & M;
	}

	ModuleInfo::~ModuleInfo()
	{}

  //FIXME
	/*const llvm::Type * ModuleInfo::lookUpType(const std::string & name) const
	{
		return getModule()->getTypeSymbolTable().lookup(name);
	}*/

}
