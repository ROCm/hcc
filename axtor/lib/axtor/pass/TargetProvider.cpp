/*
 * TargetProvider.cpp
 *
 *  Created on: 25.04.2010
 *      Author: gnarf
 */

#include <axtor/pass/TargetProvider.h>
#include <axtor/util/llvmDebug.h>

namespace axtor {
	char TargetProvider::ID = 0;

	llvm::RegisterPass<TargetProvider> __regTargetProvider("provider", "axtor - target backend and module information pass",
						true,
						true);



	TargetProvider::TargetProvider() :
		llvm::ImmutablePass(ID),
		backend(NULL),
		modInfo(NULL)
	{
		assert(false && "invalid ctor");
	}

	TargetProvider::TargetProvider(AxtorBackend & _backend, ModuleInfo & _modInfo) :
		llvm::ImmutablePass(ID),
		backend(&_backend),
		modInfo(&_modInfo)
	{
		assert(backend->hasValidType(modInfo) && "invalid ModuleInfo class for this backend");
#ifdef DEBUG
		std::cerr << "### INPUT MODULE ###\n";
		modInfo->dumpModule();
		std::cerr << "[EOF]\n";
#endif
	}

	 void TargetProvider::initializePass() {}

	AxtorBackend & TargetProvider::getBackend() const
	{
		assert(backend && "was not properly initialized");
		return *backend;
	}

	ModuleInfo & TargetProvider::getModuleInfo() const
	{
		assert(modInfo && "was not properly initialized");
		return *modInfo;
	}

	 void TargetProvider::getAnalysisUsage(llvm::AnalysisUsage & usage) const
	{
		getBackend().getAnalysisUsage(usage);
	}

	 BlockCopyTracker & TargetProvider::getTracker() const
	 {
		 assert(modInfo && "was not properly initialized");
		 return *modInfo;
	 }
}
