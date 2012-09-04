/*
 * CompilerLog.cpp
 *
 *  Created on: 25.04.2010
 *      Author: gnarf
 */
#include <axtor/console/CompilerLog.h>
#include <assert.h>
#include <stdlib.h>

#ifdef DEBUG
#include <sstream>
#endif

namespace axtor {
	llvm::raw_ostream * CompilerLog::msgStream = NULL;

	void CompilerLog::assertStream()
	{
		assert(msgStream && "MessageWriter::msgStream was not initialized!");
	}

	void CompilerLog::terminate()
	{
#ifdef DEBUG
		assert(false);
#endif
		std::cerr << "aborting . . .";
		abort();
	}

	void CompilerLog::init(llvm::raw_ostream & _msgStream)
	{
		msgStream = &_msgStream;
	}

	/*
	 * print a warning
	 */
	void CompilerLog::warn(const llvm::Value * value, const std::string & msg)
	{
		assertStream();

		(*msgStream) << "\n----\n";
		value->print(*msgStream, NULL);
		(*msgStream) << "\n\twarning: " << msg << '\n';
	}

	void CompilerLog::warn(const llvm::Type * type, const std::string & msg)
	{
		assertStream();

		(*msgStream) << "\n----\n";
		type->print(*msgStream);
		(*msgStream) << "\n\twarning: " << msg << '\n';
	}

	void CompilerLog::warn(const std::string & msg)
	{
		assertStream();

		(*msgStream) << "\n----\n";
		(*msgStream) << "\n\twarning: " << msg << '\n';
	}

	/*
	 * print a warning if @cond is not satifisfied and quit
	 */
	void CompilerLog::fail(const llvm::Value * value, const std::string & msg)
	{
		assertStream();

		(*msgStream) << "\n----\n";
		value->print(*msgStream, NULL);
		(*msgStream) << "\n\terror: " << msg << '\n';
		terminate();
	}

	void CompilerLog::fail(const llvm::Function * func, const std::string & msg)
	{
		assertStream();

		(*msgStream) << "\n----\n";
		func->dump();
		(*msgStream) << "\n\terror: " << msg << '\n';
		terminate();
	}


	void CompilerLog::fail(const llvm::Type * type, const std::string & msg)
	{
		assertStream();

		(*msgStream) << "\n----\n";
		type->print(*msgStream);
		(*msgStream) << "\n\terror: " << msg << '\n';
		terminate();
	}

	void CompilerLog::fail(const std::string & msg)
	{
		assertStream();

		(*msgStream) << "\n----\n";
		(*msgStream) << "\n\terror: " << msg << '\n';
		terminate();
	}
}
