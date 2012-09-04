/*
 * OCLModuleInfo.cpp
 *
 *  Created on: 25.04.2010
 *      Author: gnarf
 */

#include <axtor_ocl/OCLModuleInfo.h>

#include <axtor/util/llvmDebug.h>

#include "llvm/Metadata.h"

namespace axtor {

bool OCLModuleInfo::scanForDoubleType()
{
	return true;
}

bool OCLModuleInfo::requiresDoubleType()
{
	return usesDoubleType;
}

OCLModuleInfo::OCLModuleInfo(llvm::Module * mod,
                             std::vector<llvm::Function*> kernels, 
                             std::ostream & out) : ModuleInfo(*mod), 
                                                   mod(mod), 
                                                   kernels(kernels), 
                                                   out(out) {
	assert(mod && "no module specified");
	usesDoubleType = scanForDoubleType();
}

/*
 * helper method for creating a ModuleInfo object from a module and a bind file
 */
OCLModuleInfo OCLModuleInfo::createTestInfo(llvm::Module *mod, 
                                            std::ostream &out)
{
	//llvm::Function * kernelFunc = mod->getFunction("compute");
  //assert(kernelFunc && "could not find \" compute \" - function");
  std::vector<llvm::Function*> kernels; 
  llvm::NamedMDNode *openCLMetadata = mod->getNamedMetadata("opencl.kernels");
  assert(openCLMetadata && "No kernels in the module");

  for(unsigned K = 0, E = openCLMetadata->getNumOperands(); K != E; ++K) {
    llvm::MDNode &kernelMD = *openCLMetadata->getOperand(K);
    kernels.push_back(llvm::cast<llvm::Function>(kernelMD.getOperand(0)));
  }

	return OCLModuleInfo(mod, kernels, out);
}

std::ostream & OCLModuleInfo::getStream()
{
	return out;
}

bool OCLModuleInfo::isKernelFunction(llvm::Function *function)
{
  if(!function)
    return false;
  llvm::NamedMDNode *openCLMetadata = mod->getNamedMetadata("opencl.kernels");
  if(!openCLMetadata)
    return false;

  for(unsigned K = 0, E = openCLMetadata->getNumOperands(); K != E; ++K) {
    llvm::MDNode &kernelMD = *openCLMetadata->getOperand(K);
    if(kernelMD.getOperand(0) == function)
      return true;
  }
  return false;
}

std::vector<llvm::Function*> OCLModuleInfo::getKernelFunctions()
{
	return kernels;
}

void OCLModuleInfo::dump()
{
	std::cerr << "OpenCL shader module descriptor\n"
			      << "kernel functions:\n";
  for(std::vector<llvm::Function*>::const_iterator kernel = kernels.begin(), 
      end = kernels.end();
      kernel != end; ++kernel)
    std::cerr << "* " << (*kernel)->getName().str() << "\n";
}

void OCLModuleInfo::dumpModule()
{
	mod->dump();
}

IdentifierScope OCLModuleInfo::createGlobalBindings()
{
	ConstVariableMap globals;

	for(llvm::Module::const_global_iterator it = mod->global_begin(); it != mod->global_end(); ++it)
	{
		if (llvm::isa<llvm::GlobalVariable>(it))
		{
			std::string name = it->getName().str();
			globals[it] = VariableDesc(it, name);
		}
	}

#ifdef DEBUG
	std::cerr << "ModuleContext created\n";
#endif

	return IdentifierScope(globals);
}

/*
 * checks whether all types are supported and
 */
void OCLModuleInfo::verifyModule()
{
/*		for (llvm::Module::iterator func = getModule()->begin(); func != getModule()->end(); ++func)
	{
		//check function arguments first
		for (llvm::Function::iterator block = func->begin(); block != func->end(); ++block)
		{
			for(llvm::BasicBlock::iterator inst = block->begin(); inst != block->end; ++block) {

			}

		}
	} */
	axtor::verifyModule(*mod);
}

bool OCLModuleInfo::isTargetModule(llvm::Module * other) const
{
	return mod == other;
}

/*llvm::TypeSymbolTable & OCLModuleInfo::getTypeSymbolTable()
{
	return mod->getTypeSymbolTable();
}*/

llvm::Module * OCLModuleInfo::getModule()
{
	return mod;
}

void OCLModuleInfo::runPassManager(llvm::PassManager & pm)
{
	pm.run(*getModule());
}

}
