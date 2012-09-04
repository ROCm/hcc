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
 * LoopParser.h
 *
 *  Created on: 20.02.2011
 *      Author: Simon Moll
 */

#ifndef LOOPPARSER_HPP_
#define LOOPPARSER_HPP_

#include <axtor/parsers/PrimitiveParser.h>
#include <axtor/console/CompilerLog.h>

namespace axtor {

	/*
	 * Infinite Loop Parser - will abort when applied to multi-exit loops (does not return a solver procedure)
	 *
	 * TODO abort on irreducible loops
	 */
	class LoopParser : public PrimitiveParser
	{
		static LoopParser instance;

	public:
		class LoopBuilderSession : public PrimitiveParser::BuilderSession
		{
public:
			LoopBuilderSession(RegionVector regions, llvm::BasicBlock * entry, llvm::BasicBlock * requiredExit);

			virtual RestructuringProcedure * getSolver() const;

			virtual ast::ControlNode * build(ast::NodeMap children, llvm::BasicBlock * exitBlock);

			virtual std::string getName() const;

			virtual void dump();
		};

		virtual BuilderSession * tryParse(llvm::BasicBlock * entry, ExtractorContext context, AnalysisStruct & analysis);

		static LoopParser * getInstance();
	};
}


#endif /* LOOPPARSER_HPP_ */
