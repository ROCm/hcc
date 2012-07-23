#!/usr/bin/env python
import os

cwd = os.getcwd()
synDir = "../../../../sync_clangsvn/clangsvn"
print "Change directory to ../../../../sync_clangsvn/clangsvn"
os.chdir(synDir)

print "Run: hg pull"
os.system("hg pull")

print "Change directory back to clang"
os.chdir(cwd)

print "Run: hg update default"
os.system("hg update default")

print "Run: hg pull %s" % synDir
os.system("hg pull " + synDir)

print "Run: hg update"
os.system("hg update")

print "Run: hg update cppamp"
os.system("hg update cppamp")

print "(you need run 'hg push' to push changes)"
