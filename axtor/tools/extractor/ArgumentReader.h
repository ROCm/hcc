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

#ifndef _ARGUMENTREADER_HPP
#define _ARGUMENTREADER_HPP

class ArgumentReader
{
	int argc;
	char ** argv;

public:
	ArgumentReader(int _argc, char ** _argv) :
		argc(_argc),
		argv(_argv)
	{}

	int getNumArgs()
	{
		return argc - 1;
	}

	std::string getCommand()
	{
		return argv[0];
	}

	std::string get(int idx)
	{
		return argv[idx + 1];
	}

	bool readOption(std::string name, int len, std::vector<std::string> & oParams)
	{
		int start = 0;

		oParams.clear();
		for(;start < getNumArgs(); ++start)
		{
			if (get(start) == name)
			{
				int base = start + 1;

				for(int paramOff = 0; paramOff < len; ++paramOff)
				{
					int idx = base + paramOff;

					if (idx < getNumArgs()) {
						oParams.push_back( get(idx) );
					} else {
						return false;
					}
				}

				return true;
			}
		}

		return false;
	}

	bool readOption(std::string name, std::string & oParam)
	{
		std::vector<std::string> params;
		if (readOption(name, 1, params)) {
			oParam = *params.begin();
			return true;
		}
		return false;
	}
};


#endif
