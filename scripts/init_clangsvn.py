#!/usr/bin/env python
import os

os.mkdir("sync_clangsvn")
os.chdir("sync_clangsvn")

# Enable hgsubversion extension
os.system("hg clone http://bitbucket.org/durin42/hgsubversion")
hgrc = open('hgrc', 'w')
hgrc.write("[extensions]\n")
hgrc.write("hgsubversion=\"hgsubversion\"\n")

# Clone clang trunk
os.system("hg clone http://llvm.org/svn/llvm-project/cfe/trunk clangsvn")
