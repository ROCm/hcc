#include <axtor/util/llvmDebug.h>

std::string axtor::toString(const BlockSet & blocks)
{
	std::stringstream out;
	out << "BlockSet {\n";
	for(BlockSet::const_iterator itBlock = blocks.begin(); itBlock != blocks.end(); ++itBlock)
	{
		llvm::BasicBlock * block = *itBlock;
		out << "\t" << (block ? block->getName().str() : "null") << ",\n";
	}
	out << "}\n";
	return out.str();
}

void axtor::dumpBlockSet(const BlockSet & blocks)
{
	std::cerr << toString(blocks) << "\n";
}

void axtor::dumpTypeSet(const TypeSet & types)
{
	std::cerr << "TypeSet {\n";
	for(TypeSet::const_iterator itType = types.begin(); itType != types.end(); ++itType)
	{
		const llvm::Type * type = *itType;
		std::cerr << "\t"; type->dump();
	}
	std::cerr << "}\n";
}

void axtor::dumpBlockVector(const BlockVector & blocks)
{
	std::cerr << "BlockVector [\n";
	for(BlockVector::const_iterator itBlock = blocks.begin(); itBlock != blocks.end(); ++itBlock)
	{
		llvm::BasicBlock * block = *itBlock;
		std::cerr << "\t" << (block ? block->getName().str() : "null") << ",\n";
	}
	std::cerr << "]\n";
}



void axtor::verifyModule(llvm::Module & mod)
{
	std::string errorMessage;
	if (llvm::verifyModule(mod, llvm::PrintMessageAction, &errorMessage)) {

            llvm::errs() << "\n\n\n##### BROKEN MODULE #####\n";
	    mod.dump();
	    llvm::errs() << "##### END OF DUMP ##### \n\n\n";
	    abort();
       }
}

void axtor::dumpUses(llvm::Value * val)
{
	std::cerr << "dumping value uses for "; val->dump();
	std::cerr << "----- begin of user list ----\n";
	for(llvm::Value::use_iterator use = val->use_begin(); use != val->use_end(); ++use)
	{
		use->dump();
	}
	std::cerr << "----- end of user list -----\n";
}
