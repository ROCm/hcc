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
 * TextStream.h
 *
 *  Created on: 03.05.2010
 *      Author: Simon Moll
 */

#ifndef TEXTSTREAM_HPP_
#define TEXTSTREAM_HPP_

#include <string>
#include <iostream>

namespace axtor {

class TextStream
{
public:
	virtual void put(std::string text)=0;
	virtual void putLine(std::string text)=0;
};

class TextFileStream : public TextStream
{
	std::ostream & out;

public:
	TextFileStream(std::ostream & _out);
	virtual void put(std::string text);
	virtual void putLine(std::string text);
};

class MultiStream : public TextStream
{
	TextStream & first;
	TextStream & second;

public:
	MultiStream(TextStream & _first, TextStream & _second);
	virtual void put(std::string);
	virtual void putLine(std::string);
};

}
#endif /* TEXTSTREAM_HPP_ */
