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
 * IfParser.h
 *
 *  Created on: 20.02.2011
 *      Author: Simon Moll
 */

#ifndef IFPARSER_HPP_
#define IFPARSER_HPP_

#include "PrimitiveParser.h"
#include <axtor/ast/BasicNodes.h>
#include <axtor/util/llvmShortCuts.h>

#include <axtor/solvers/NodeSplittingRestruct.h>

namespace axtor {
	/*
	 * combined IF and IF..ELSE parser class
	 *
	 * tryParse will return IF of IF..ELSE Builder Sessions depending on what kind of control-flow pattern was detected
	 *
	 * uses the Node-Splitting Solver procedure
	 */
	class IfParser : public PrimitiveParser
	{
		static IfParser instance;

	public:
		/*
		 * IF - builder (w/o ELSE)
		 */
		class IfBuilderSession : public PrimitiveParser::BuilderSession
		{
		public:
			IfBuilderSession(RegionVector regions, llvm::BasicBlock * exitBlock, llvm::BasicBlock * entryBlock);

			ast::ControlNode * build(ast::NodeMap children, llvm::BasicBlock * exitBlock);

			RestructuringProcedure * getSolver() const;

			virtual std::string getName() const;

			virtual void dump();
		};

		/*
		 * IF..ELSE builder
		 */
		class IfElseBuilderSession : public PrimitiveParser::BuilderSession
		{
		public:
			IfElseBuilderSession(RegionVector regions, llvm::BasicBlock * exitBlock, llvm::BasicBlock * entryBlock);

			ast::ControlNode * build(ast::NodeMap children, llvm::BasicBlock * exitBlock);

			RestructuringProcedure * getSolver() const;

			virtual std::string getName() const;

			virtual void dump();
		};

		/*
		 * tries to set up a builder session for IF primitives and falls back to IF..ELSE
		 */
		virtual BuilderSession * tryParse(llvm::BasicBlock * entry, ExtractorContext context, AnalysisStruct & analysis);

		static IfParser * getInstance();
	};
}

#endif /* IFPARSER_HPP_ */
