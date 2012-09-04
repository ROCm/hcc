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
 * extractor.cpp
 *
 *  Created on: 07.02.2010
 *      Author: Simon Moll
 */

#include <llvm/PassManagers.h>
#include <llvm/Target/TargetData.h>

#include <iostream>
#include <stdio.h>
#include <fstream>

#include <axtor/config.h>

#include <axtor/metainfo/ModuleInfo.h>
#include <axtor/backend/AxtorBackend.h>

#ifdef ENABLE_OPENCL
#include <axtor_ocl/OCLModuleInfo.h>
#include <axtor_ocl/OCLBackend.h>
#endif

#include <axtor/util/llvmTools.h>
#include <axtor/console/CompilerLog.h>
#include <axtor/Axtor.h>

#include "ArgumentReader.h"

static void dumpHelp();


#ifdef ENABLE_OPENCL
static int run_OpenCL(ArgumentReader args)
{
	std::ostream * outStream = NULL;

	if (args.getNumArgs() > 0) {
		std::string inputFile = args.get(0);
		llvm::Module * mod = axtor::createModuleFromFile(inputFile);

		if (!mod) {
			axtor::Log::fail("no input module specified!");
		}


		axtor::StringVector params;
		if (args.readOption("-o", 1, params)) {
			std::string outFile = params.back();
			outStream = new std::ofstream(outFile.c_str(), std::ios::out);
		}

		axtor::OCLBackend backend;

		if (outStream) {
			axtor::OCLModuleInfo modInfo = axtor::OCLModuleInfo::createTestInfo(mod, *outStream);
			axtor::translateModule(backend, modInfo);
			delete outStream;

		} else {
#ifdef EVAL_DECOMPILE_TIME
			std::stringstream stream;
			axtor::OCLModuleInfo modInfo = axtor::OCLModuleInfo::createTestInfo(mod, stream);
#else
			axtor::OCLModuleInfo modInfo = axtor::OCLModuleInfo::createTestInfo(mod, std::cout);
#endif
			axtor::translateModule(backend, modInfo);
		}
		return 0;
	}

	std::cerr << "no input file specified!" << std::endl;
	dumpHelp();
	return -1;
}

static void dump_OpenCL()
{
	std::cerr << "Options for the OpenCL Backend (-m OCL)"
			  << "\n"
			  << "<infile> -o <outputFile>"
			  << "\n"
			  << "-o <FILE>  output file"
			  << std::endl;
}
#endif

static void dumpHelp()
{
#ifdef ENABLE_OPENCL
	dump_OpenCL();
#endif
}



int main(int argc, char ** argv)
{
#ifdef EVAL_DECOMPILE_TIME
	std::cerr << "(!!!) EVALUATION BUILD (!!!)\n";
#endif

	axtor::initialize(true);

	ArgumentReader args(argc, argv);
	axtor::StringVector backendVector;

	if (args.readOption("-m", 1, backendVector)) {
		std::string backendStr = backendVector.back();

		if (false) {} //just a dummy
#ifdef ENABLE_OPENCL
		else if (backendStr == "OCL")
			return run_OpenCL(args);
#endif
	}

	std::cerr << "no backend specified\n";
	dumpHelp();
	return -1;
}
