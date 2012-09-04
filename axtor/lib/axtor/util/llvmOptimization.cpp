/*
 * llvmOptimization.cpp
 *
 *  Created on: Aug 25, 2011
 *      Author: Simon Moll
 */

#include <axtor/util/llvmOptimization.h>

#include <llvm/Target/TargetData.h>
#include <llvm/Module.h>
#include <llvm/DerivedTypes.h>
#include <llvm/Instructions.h>
#include <llvm/LinkAllPasses.h>
#include <llvm/PassManager.h>
#include <llvm/Linker.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Analysis/Passes.h>

namespace axtor {

static inline void createStandardModulePasses(llvm::PassManagerBase *PM, int threshold) {
	assert(false && "dont optimise with builtins!!");
  //FIXME
	//llvm::createStandardAliasAnalysisPasses(PM);

  // Basic AliasAnalysis support.
  // Add TypeBasedAliasAnalysis before BasicAliasAnalysis so that
  // BasicAliasAnalysis wins if they disagree. This is intended to help
  // support "obvious" type-punning idioms.
  PM->add(llvm::createTypeBasedAliasAnalysisPass());
  PM->add(llvm::createBasicAliasAnalysisPass());





	PM->add(llvm::createGlobalOptimizerPass());     // Optimize out global vars

	PM->add(llvm::createIPSCCPPass());              // IP SCCP
	PM->add(llvm::createDeadArgEliminationPass());  // Dead argument elimination
	PM->add(llvm::createInstructionCombiningPass());  // Clean up after IPCP & DAE
    PM->add(llvm::createCFGSimplificationPass());     // Clean up after IPCP & DAE

    // Start of CallGraph SCC passes.
	PM->add(llvm::createPruneEHPass());             // Remove dead EH info
	PM->add(llvm::createFunctionAttrsPass());       // Set readonly/readnone attrs
	PM->add(llvm::createArgumentPromotionPass());   // Scalarize uninlined fn args

    // Start of function pass.
    // Break up aggregate allocas, using SSAUpdater.
    PM->add(llvm::createScalarReplAggregatesPass(threshold, false));
    PM->add(llvm::createEarlyCSEPass());              // Catch trivial redundancies
    PM->add(llvm::createJumpThreadingPass());         // Thread jumps.
    PM->add(llvm::createCorrelatedValuePropagationPass()); // Propagate conditionals
    PM->add(llvm::createCFGSimplificationPass());     // Merge & remove BBs
    PM->add(llvm::createInstructionCombiningPass());  // Combine silly seq's

    PM->add(llvm::createTailCallEliminationPass());   // Eliminate tail calls
    PM->add(llvm::createCFGSimplificationPass());     // Merge & remove BBs
    PM->add(llvm::createReassociatePass());           // Reassociate expressions
    PM->add(llvm::createLoopRotatePass());            // Rotate Loop
    PM->add(llvm::createLICMPass());                  // Hoist loop invariants
    PM->add(llvm::createLoopUnswitchPass(false));
    PM->add(llvm::createInstructionCombiningPass());
    PM->add(llvm::createIndVarSimplifyPass());        // Canonicalize indvars
    PM->add(llvm::createLoopIdiomPass());             // Recognize idioms like memset.
    PM->add(llvm::createLoopDeletionPass());          // Delete dead loops
    PM->add(llvm::createInstructionCombiningPass());  // Clean up after the unroller
      PM->add(llvm::createGVNPass());                 // Remove redundancies
    PM->add(llvm::createMemCpyOptPass());             // Remove memcpy / form memset
    PM->add(llvm::createSCCPPass());                  // Constant prop with SCCP

    // Run instcombine after redundancy elimination to exploit opportunities
    // opened up by them.
    PM->add(llvm::createInstructionCombiningPass());
    PM->add(llvm::createJumpThreadingPass());         // Thread jumps
    PM->add(llvm::createCorrelatedValuePropagationPass());
    PM->add(llvm::createDeadStoreEliminationPass());  // Delete dead stores
    PM->add(llvm::createAggressiveDCEPass());         // Delete dead instructions
    PM->add(llvm::createCFGSimplificationPass());     // Merge & remove BBs

	PM->add(llvm::createStripDeadPrototypesPass()); // Get rid of dead prototypes
	//PM->add(llvm::createDeadTypeEliminationPass()); // Eliminate dead types

	// GlobalOpt already deletes dead functions and globals, at -O3 try a
	// late pass of GlobalDCE.  It is capable of deleting dead cycles.
	PM->add(llvm::createGlobalDCEPass());         // Remove dead fns and globals.

	PM->add(llvm::createConstantMergePass());       // Merge dup global constants
  }


	void optimizeModule(llvm::Module * module)
	{

		//Run Standard Passes

		//module passes
		llvm::PassManager MPM;
		llvm::TargetData * td = new llvm::TargetData(module->getDataLayout());
		MPM.add(td);
		createStandardModulePasses(&MPM, 32);
		MPM.add(llvm::createGlobalDCEPass());
		MPM.add(llvm::createStripDeadPrototypesPass());
		//MPM.add(llvm::createDeadTypeEliminationPass());
		MPM.add(llvm::createScalarReplAggregatesPass(32, false));

		MPM.run(*module);
	}
}
