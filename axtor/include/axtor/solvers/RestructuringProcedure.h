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
 * RestructuringProcedure.h
 *
 *  Created on: 20.02.2011
 *      Author: Simon Moll
 */

#ifndef RESTRUCTURINGPROCEDURE_HPP_
#define RESTRUCTURINGPROCEDURE_HPP_

#include <axtor/CommonTypes.h>

#include <axtor/util/AnalysisStruct.h>
#include <axtor/util/ExtractorRegion.h>

namespace axtor
{
	/*
	 * Interface for restructuring procedures for acyclic control-flow
	 *
	 * (Implement as Singleton)
	 */

	class RestructuringProcedure
	{
	public:
		virtual ~RestructuringProcedure() {};
		/*
		 * converts the @regions into valid regions with respect to the single exit node criterion of acyclic abstract high-level nodes
		 * 	oExitBlock
		 * 		requiredExit (if defined) or a newly generated exit block otw. (if any)
		 * returns
		 * 		if the CFG was modified and needs to be reparsed
		 */
		virtual bool resolve(RegionVector & regions, llvm::BasicBlock * requiredExit, const ExtractorContext & context, AnalysisStruct & analysis, llvm::BasicBlock *& oExitBlock) = 0;
	};
}

#endif /* RESTRUCTURINGPROCEDURE_HPP_ */
