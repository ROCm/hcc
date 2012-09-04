/*
 * stringutil.cpp
 *
 *  Created on: 25.04.2010
 *      Author: gnarf
 */

#include <axtor/util/stringutil.h>
#include <axtor/console/CompilerLog.h>

namespace axtor {


template<typename T>
std::string str(T number)
{
	std::ostringstream out;
	out << number;
	return out.str();
}

template<>
std::string str(float number)
{
	std::ostringstream out;
	out << number;

	std::string tmp = out.str();

	if (tmp.find("e") != std::string::npos) { //scientific notation
		return tmp + "f";
	} else	if (tmp.find(".") == std::string::npos) { //pure decimal
		return tmp + ".0f";
	} else {
		return tmp + "f";
	}
}

template<>
std::string str(double number)
{
	std::ostringstream out;
	out << number;

	std::string tmp = out.str();

	if (tmp.find("e") != std::string::npos) { //scientific notation
		return tmp + "f";
	} else	if (tmp.find(".") == std::string::npos) { //pure decimal
		return tmp + ".0f";
	} else {
		return tmp + "f";
	}
}

template std::string str<uint64_t>(uint64_t number);
template std::string str<unsigned int>(unsigned int number);
template std::string str<int>(int number);
template std::string str<long>(long number);
template std::string str<char>(char number);


bool isNum(char c)
{
   return ('0' <= c) && (c <= '9');
}

bool isABC(char c)
{
   return (('a' <= c) && (c <= 'z')) ||
          (('A' <= c) && (c <= 'Z'));
}

bool isAlphaNum(char c)
{
   return
      isNum(c) ||
      isABC(c);
}

int parseInt(char * cursor, char ** out)
{
	int num = 0;

	for(;*cursor != '\0' && isNum(*cursor); ++cursor)
	{
		num = 10 * num + (*cursor - '0');
	}

	*out = cursor;
	return num;
}

/*
 * returns the hex string representation for this integer (so far only for 0 <= i <= 15)
 */
std::string hexstr(int i)
{
	if (0 <= i && i < 10)
	{
		return str<int>(i);
	} else if (i > 9 && i < 16) {
		char c = 'A' + (i - 10);
		std::ostringstream ss;
		ss << c;
		return ss.str();
	} else {
		Log::fail("not implemented");
	}
}

/*
 * returns the hex-representation of an integer
 */

std::string convertToHex(uint64_t num, unsigned int digits)
{
	assert(sizeof(uint64_t) * 2 >= (digits) && "read exceeds number of bytes");

	std::string hex(digits, '0');

	for(unsigned int i = 0; i < digits; ++i)
	{
		int charIdx = digits - i - 1;
		unsigned char bits = (num >> (4 * charIdx)) & 0xF;

		if (bits < 10)
			hex[i] = '0' + bits;
		else
			hex[i] = 'A' + (bits - 10);
	}

	return hex;
}

/*
 * returns a StringVector of tokens from @src delimited by @delimiter
 * such that there is a string before and after each occurrence of @delimiter (even if it's an empty string)
 */
StringVector tokenizeIsolated(std::string src, std::string delimiter)
{
	typedef std::string::size_type Position;

	std::vector<std::string> result;
	Position pos = 0;
	Position nextStart = 0;
	Position start;
  //FIXME
	//bool lastToken = false;

	do
	{
		start = nextStart;
		pos = src.find_first_of(delimiter, nextStart);

		//determine @nextStart and fix @pos
		if (pos == std::string::npos) {
			pos = src.length();
			//lastToken = true;
			nextStart = pos;

		} else {
			nextStart = pos + delimiter.length();
		}

		Position len = pos - start;
		std::string token = src.substr(start, len);
#ifdef DEBUG
		std::cerr << "push:" << token << "\n";
#endif
		result.push_back(token);

	}
	while (pos < src.length());

	return result;
}
}
