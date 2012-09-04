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
 * CGIPass.h
 *
 *  Created on: Apr 12, 2011
 *      Author: Simon Moll
 */

#ifndef CGIPASS_HPP_
#define CGIPASS_HPP_

#include <axtor/config.h>

#include <llvm/Pass.h>
#include <llvm/Module.h>
#include <llvm/Analysis/CallGraph.h>

/*
 * (generic) Code Generator Intrinsics Pass
 *
 * provides implementations for the code-generator intrinsics
 */


namespace axtor {

class CGIPass : public llvm::ModulePass
{

	class Session
	{
		llvm::Module & M;
		bool removeIfUsed(const std::string & funcName);

	public:
		Session(llvm::Module & _M) :
			M(_M)
		{}

		bool run();

		/*
		 * implements llvm.memcpy for GPUs
		 */
		void lowerMemCpy();

		void lowerMemSet();

		static const char * getPassName() { return "CGIPass - OCL Code Generator Intrinsics"; }
	};

	public:
		static char ID;

		CGIPass();

		virtual void getAnalysisUsage(llvm::AnalysisUsage & usage) const;

		bool runOnModule(llvm::Module & M);

		virtual const char * getPassName() const
		{
			return Session::getPassName();
		}
};

}

#endif /* CGIPASS_HPP_ */
