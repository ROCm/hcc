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
 * CompilerLog.h
 *
 *  Created on: 01.03.2010
 *      Author: Simon Moll
 */

#ifndef COMPILERLOG_HPP_
#define COMPILERLOG_HPP_

#include <stdlib.h>
#include <iostream>

#include <llvm/Type.h>
#include <llvm/Value.h>
#include <llvm/Function.h>
#include <llvm/Support/raw_ostream.h>

#include <axtor/Annotations.h>

namespace axtor {

/*
 * default interface for communicating compiler warnings/errors
 */
class CompilerLog
{
private:
	static llvm::raw_ostream * msgStream;

	static void assertStream();

	static void terminate() ANNOT_NORETURN;

public:

	static void init(llvm::raw_ostream & _msgStream);

	/*
	 * print the @msg-warning and continue
	 */
	static void warn(const llvm::Value * value, const std::string & msg);

	static void warn(const llvm::Type * type, const std::string & msg);

	static void warn(const std::string & msg);

	/*
	 * print the error message @msg, dump the object and quit
	 */
	static void fail(const llvm::Function * Function, const std::string & msg) ANNOT_NORETURN;

	static void fail(const llvm::Value * value, const std::string & msg) ANNOT_NORETURN;

	static void fail(const llvm::Type * type, const std::string & msg) ANNOT_NORETURN;

	static void fail(const std::string & msg) ANNOT_NORETURN;
};

typedef CompilerLog Log;
}


#endif /* MESSAGEWRITER_HPP_ */
