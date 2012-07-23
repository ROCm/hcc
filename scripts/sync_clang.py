#!/usr/bin/env python
import os

cwd = os.getcwd()
synDir = "../../../../sync_clangsvn/clangsvn"
print "Change directory to ../../../../sync_clangsvn/clangsvn"
os.chdir(synDir)

print "Do: hg pull"
os.system("hg pull")

print "Change directory back to clang"
os.chdir(cwd)

print "Do: hg update default"
os.system("hg update default")

print "Do: hg pull %s" % synDir
os.system("hg pull " + synDir)

print "Do: hg update"
os.system("hg update")

print "Do: hg update cppamp"
os.system("hg update cppamp")
