#!/usr/bin/env python

import os
import sys

cur_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
top_dir = os.path.abspath(cur_dir + os.sep + "..")

llvm_dir = top_dir + os.sep + "compiler"
os.chdir(llvm_dir)
os.system("svn update")

clang_dir = llvm_dir + os.sep + "tools" + os.sep + "clang"
os.chdir(clang_dir)
os.system("hg pull https://bitbucket.org/multicoreware/cppamp")

