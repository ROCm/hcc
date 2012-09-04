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
 * stringutil.h
 *
 *  Created on: 08.02.2010
 *      Author: Simon Moll
 */

#ifndef STRINGUTIL_HPP_
#define STRINGUTIL_HPP_

#include <string>
#include <vector>
#include <stdio.h>
#include <sstream>
#include <iostream>
#include <assert.h>
#include <stdint.h>

typedef std::vector<std::string> StringVector;

namespace axtor {

template<typename T>
std::string str(T number);

template<>
std::string str(float number);

bool isNum(char c);

bool isABC(char c);

bool isAlphaNum(char c);

int parseInt(char * cursor, char ** out);

/*
 * returns the hex string representation for this integer (so far only for 0 <= i <= 15)
 */
std::string hexstr(int i);

/*
 * returns the hex string representation of an integer variable
 */
std::string convertToHex(uint64_t num, unsigned int bytes);

/*
 * returns a StringVector of tokens from @src delimited by @delimiter
 * such that there is a string before and after each occurrence of @delimiter (even if it's an empty string)
 */
std::vector<std::string> tokenizeIsolated(std::string src, std::string delimiter);

}

#endif /* STRINGUTIL_HPP_ */
