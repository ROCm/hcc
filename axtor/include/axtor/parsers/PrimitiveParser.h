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
 * PrimitiveParser.h
 *
 *  Created on: 20.02.2011
 *      Author: Simon Moll
 */

#ifndef PRIMITIVEPARSER_HPP_
#define PRIMITIVEPARSER_HPP_

#include <axtor/CommonTypes.h>

#include <axtor/ast/ASTNode.h>
#include <axtor/util/AnalysisStruct.h>
#include <axtor/util/ExtractorRegion.h>

#include <axtor/solvers/RestructuringProcedure.h>

namespace axtor {

	class PrimitiveParser
	{
	public:
		/*
		 * primitive-dependent builder session which is set up by @super::tryParse.
		 *
		 * If all child nodes have been parsed @build is called to build a ast::ControlNode from it
		 * This is implemented in its own class, so that @build can access information gathered in @super::tryParse
		 */
		class BuilderSession
		{
			RegionVector regions; //headers to the primitives child regions
			llvm::BasicBlock * entryBlock; //entry block to the primitive
			llvm::BasicBlock * requiredExit; //the strictly required exit block of all regions (that is, if a child region exits at all)


		public:
			ExtractorRegion & getRegion(uint i) { return regions[i]; }
			RegionVector & getRegions() { return regions; }
			llvm::BasicBlock * getRequiredExit() { return requiredExit; }
			llvm::BasicBlock * getEntryBlock() { return entryBlock; }


			BuilderSession(RegionVector _regions, llvm::BasicBlock * _entryBlock, llvm::BasicBlock * _requiredExit) :
				regions(_regions), entryBlock(_entryBlock), requiredExit(_requiredExit) {}

			/*
			 * returns the solver procedure for the restructuring child regions
			 */
			virtual RestructuringProcedure * getSolver() const = 0;

			/*
			 * build a ControlNode from the child nodes
			 */
			virtual ast::ControlNode * build(ast::NodeMap children, llvm::BasicBlock * exitBlock) = 0;

			/*
			 * return a builder name (for debugging purposes)
			 */
			virtual std::string getName() const = 0;

			/*
			 * dump some debug info
			 */
			virtual void dump() = 0 ;
		};

		/*
		 * match the control-flow pattern at entry with this primitive type. If is matches, set up a BuilderSession for this primitive
		 */
		virtual BuilderSession * tryParse(llvm::BasicBlock * entry, ExtractorContext context, AnalysisStruct & analysis) = 0;
	};
}

#endif /* PRIMITIVEPARSER_HPP_ */
