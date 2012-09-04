/*
 * PredicateRestruct.h
 *
 *  Created on: 27 Feb 2012
 *      Author: v1smoll
 */

#ifndef PREDICATERESTRUCT_H_
#define PREDICATERESTRUCT_H_

#include "RestructuringProcedure.h"

#include <axtor/config.h>
#include <axtor/util/llvmShortCuts.h>
#include <axtor/util/llvmDuplication.h>

namespace axtor
{
	/*
	 * Uses enumeration to tunnel the control-flow to joined exits through a virtual dispatch block ("fused")
	 *
	 * This pass requires mem2reg: It will produce broken code, if values are used by instructions in
	 * the joined exits from predecessors that are not received via PHI-nodes.
	 *
	 */

	class PredicateRestruct : public RestructuringProcedure
	{
		static PredicateRestruct instance;

	public:

		PredicateRestruct();
		~PredicateRestruct();
		/*
		 * converts the @regions into valid regions with respect to the single exit node criterion of acyclic abstract high-level nodes
		 * returns
		 * 		requiredExit (if defined) or a newly generated exit block otw. (if any)
		 */
		bool resolve(RegionVector & regions, llvm::BasicBlock * requiredExit, const ExtractorContext & context, AnalysisStruct & analysis, llvm::BasicBlock *& oExitBlock);

		static PredicateRestruct * getInstance();
	};
}


#endif /* PREDICATERESTRUCT_H_ */
