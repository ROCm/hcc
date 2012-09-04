#include <axtor/CommonTypes.h>

namespace axtor {

	VariableDesc::VariableDesc(const llvm::Value * val, std::string _name) :
		type(val->getType()),
		name(_name)
	{
		isAlloca = llvm::isa<llvm::AllocaInst>(val);
	}

	VariableDesc::VariableDesc() :
		type(NULL),
		name("UNDEFINED"),
		isAlloca(false)
	{}

	void VariableDesc::print(llvm::raw_ostream & out)
	{
		type->print(out); out << name << '\n';
	}






	ExtractorContext::ExtractorContext() :
		parentLoop(NULL),
		continueBlock(NULL),
		breakBlock(NULL),
		exitBlock(NULL),
		isPrecheckedLoop(false)
	{}

	BlockSet ExtractorContext::getRegularExits() const
	{
		BlockSet tmp;

		if (continueBlock)
			tmp.insert(continueBlock);
		if (breakBlock)
			tmp.insert(breakBlock);

		return tmp;
	}

	BlockSet ExtractorContext::getAnticipatedExits() const
	{
		BlockSet tmp;

		if (continueBlock)
			tmp.insert(continueBlock);
		if (breakBlock)
			tmp.insert(breakBlock);
		if (exitBlock)
			tmp.insert(exitBlock);

		return tmp;
	}

	void ExtractorContext::dump() const
	{
		dump("");
	}

	void ExtractorContext::dump(std::string prefix) const
	{
		std::cerr << prefix << "Context {\n"
				  << prefix << "\tcontinueBlock = " << (continueBlock ? continueBlock->getName().str() : "null") << "\n"
				  << prefix << "\tbreakBlock    = " << (breakBlock ? breakBlock->getName().str() : "null") << "\n"
				  << prefix << "\texitBlock     = " << (exitBlock ? exitBlock->getName().str() : "null") << "\n"
			      << prefix << "\tparentLoop    = " << (!parentLoop ? "none\n" : ""); if (parentLoop) parentLoop->dump();
		std::cerr << prefix << "}\n";
	}






	IdentifierScope::IdentifierScope(const ConstVariableMap & _identifiers) :
		parent(NULL),
		identifiers(_identifiers)
	{}

	IdentifierScope::IdentifierScope(IdentifierScope * _parent, const ConstVariableMap & _identifiers) :
		parent(_parent),
		identifiers(_identifiers)
	{}

	IdentifierScope::IdentifierScope(IdentifierScope * _parent) :
		parent(_parent)
	{}

	IdentifierScope * IdentifierScope::getParent() const
	{
		return parent;
	}

	const VariableDesc * IdentifierScope::lookUp(const llvm::Value * value) const
	{
		ConstVariableMap::const_iterator it = identifiers.find(value);

		if (it != identifiers.end()) {
			return &(it->second);

		} else if (parent) {
			return parent->lookUp(value);
		} else {
			return 0;
		}
	}

	void IdentifierScope::bind(const llvm::Value * val, VariableDesc desc)
	{
		identifiers[val] = desc;
	}

	ConstVariableMap::const_iterator IdentifierScope::begin() const
	{
		return identifiers.begin();
	}

	ConstVariableMap::const_iterator IdentifierScope::end() const
	{
		return identifiers.end();
	}
}
