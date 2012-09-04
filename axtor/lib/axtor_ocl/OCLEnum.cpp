/*
 * OCLEnum.cpp
 *
 *  Created on: 25 Jan 2012
 *      Author: v1smoll
 */

#include <axtor_ocl/OCLEnum.h>

#include <axtor/util/llvmConstant.h>

bool axtor::evaluateEnum_MemFence(llvm::Value * val, std::string & result)
{
	uint64_t enumIndex;
	if (! evaluateInt(val, enumIndex))
		return false;

	switch(enumIndex)
	{
	case 0: result = "CLK_LOCAL_MEM_FENCE"; break;
	case 1: result = "CLK_GLOBAL_MEM_FENCE"; break;
	default:
		return false;
	}

	return true;
}
