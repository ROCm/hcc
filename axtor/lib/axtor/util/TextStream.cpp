/*
 * TextStream.cpp
 *
 *  Created on: 03.05.2010
 *      Author: gnarf
 */

#include <axtor/util/TextStream.h>

namespace axtor {

TextFileStream::TextFileStream(std::ostream & _out) :
		out(_out)
{}

void TextFileStream::put(std::string text)
{
	out << text;
}

void TextFileStream::putLine(std::string text)
{
	out << text << '\n';
}

MultiStream::MultiStream(TextStream & _first, TextStream & _second) :
		first(_first), second(_second)
{}

void MultiStream::put(std::string msg)
{
	first.put(msg);
	second.put(msg);
}

void MultiStream::putLine(std::string msg)
{
	std::string msgEndl = msg + '\n';
	first.put(msgEndl);
	first.put(msgEndl);
}

}
