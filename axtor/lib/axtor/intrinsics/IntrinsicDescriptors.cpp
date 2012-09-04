/*
 * IntrinsicDescriptors.cpp
 *
 *  Created on: 25.04.2010
 *      Author: gnarf
 */

#include <axtor/intrinsics/IntrinsicDescriptors.h>

#include <axtor/util/stringutil.h>

namespace axtor {

uint IntrinsicDescriptor::getSize(StringVector::const_iterator start, StringVector::const_iterator end)
{
	uint size = 0;
	for(; start != end; ++start, ++size) continue;

	return size;
}

/*
 * class for function style intrinsics
 */

IntrinsicFuncDesc::IntrinsicFuncDesc(std::string _funcName, uint _numArgs) :
	funcName(_funcName),
	numArgs(_numArgs)
{}

std::string IntrinsicFuncDesc::build(StringVector::const_iterator start, StringVector::const_iterator end)
{
	uint size = end - start;

	if (size != numArgs) {
		Log::fail(funcName + " intrinsic called with illegal argument count. expected " + str<uint>(numArgs) + " was " + str<uint>(size));
	}

	std::string list = "(";

	for(StringVector::const_iterator it = start; it != end; ++it)
	{
		if (it != start) {
			list += ',' + *it;
		} else {
			list += *it;
		}
	}

	list+= ')';

	return funcName + list;
}

	/*
	 * intrinsic resulting in an assignment
	 */
std::string IntrinsicAssignmentDesc::build(StringVector::const_iterator start, StringVector::const_iterator end)
{
	assert(getSize(start, end) == 2 && "creates a simple assignment");
	return start[0] + " = " + start[1];
}

	/*
	 * infix operator intrinsic
	 */
IntrinsicInfixDesc::IntrinsicInfixDesc(std::string _op) : op(_op) {}

std::string IntrinsicInfixDesc::build(StringVector::const_iterator start, StringVector::const_iterator end)
{
	assert(getSize(start, end) == 2 && "binary infix operator");
	return "(" + start[0] + " " + op + " " + start[1] + ")";
}

	/*
	 * unary operator intrinsic
	 */
IntrinsicUnaryDesc::IntrinsicUnaryDesc(std::string _op) : op(_op) {}

std::string IntrinsicUnaryDesc::build(StringVector::const_iterator start, StringVector::const_iterator end)
{
	assert(getSize(start, end) == 1 && "unary operator");
	return op + "(" + start[0] + ")";
}

std::string IntrinsicNopDesc::build(StringVector::const_iterator start, StringVector::const_iterator end)
{
	return "";
}

std::string IntrinsicPassDesc::build(StringVector::const_iterator start, StringVector::const_iterator end)
{
	assert(getSize(start, end) == 1 && "unary pass through intrinsic");
	return *start;
}


IntrinsicGlobalDesc::IntrinsicGlobalDesc(std::string _global) :
		global(_global)
{}

std::string IntrinsicGlobalDesc::build(StringVector::const_iterator start, StringVector::const_iterator end)
{
	return global;
}

std::string IntrinsicGetterDesc::build(StringVector::const_iterator start, StringVector::const_iterator end)
{
	std::ostringstream builder;

	builder << *start;

	for(StringVector::const_iterator itIndex = start + 1;
			itIndex != end;
			++itIndex
		)
	{
		builder << "[" << *itIndex << "]";
	}

	return builder.str();
}

std::string IntrinsicSetterDesc::build(StringVector::const_iterator start, StringVector::const_iterator end)
{
	std::ostringstream builder;

	builder << *start;

	for(StringVector::const_iterator itIndex = start + 1;
			itIndex != (end - 1);
			++itIndex
		)
	{
		builder << "[" << *itIndex << "]";
	}

	builder << " = " << *(end - 1);

	return builder.str();
}

IntrinsicCallReplacementDesc::IntrinsicCallReplacementDesc(std::string _callText) :
	callText(_callText)
{}


std::string IntrinsicCallReplacementDesc::build(StringVector::const_iterator start, StringVector::const_iterator end)
{
	assert(start == end);
	return callText;
}

IntrinsicComplexDesc::IntrinsicComplexDesc(std::string formatString)
{
	std::stringstream cleanStream;
	//remove '"' from the string
	std::string::const_iterator startFormat = formatString.begin();
	std::string::const_iterator endFormat = formatString.end();
	for(std::string::const_iterator it = startFormat; it != endFormat; ++it)
	{
		if (*it != '\"' && *it != ' ' && *it != '\t') {
			cleanStream << *it;
		}
	}

	chunks = tokenizeIsolated(cleanStream.str(), "%");
}


std::string IntrinsicComplexDesc::build(StringVector::const_iterator paramStart, StringVector::const_iterator paramEnd)
{

	std::stringstream stream;

	StringVector::const_iterator chunkStart = chunks.begin();
	StringVector::const_iterator chunkEnd= chunks.end();
	StringVector::const_iterator itChunk;
	StringVector::const_iterator itParam = paramStart;

	stream << *chunkStart;
	for(
			itChunk = chunkStart + 1;
			itChunk != chunkEnd;
			itParam++, itChunk++)
	{
		if (itParam == paramEnd) {
			Log::fail("too few parameters given in complex intrinsic call!");
		}
		stream << *itParam << *itChunk;
	}

	if (itParam != paramEnd) {
		Log::fail("too many parameters given in complex intrinsic call!");
	}

	return stream.str();
}


}
