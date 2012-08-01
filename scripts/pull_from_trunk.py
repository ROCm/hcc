#!/usr/bin/env python

import os
import sys

cur_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
top_dir = os.path.abspath(cur_dir + os.sep + "..")
clang_dir = top_dir + os.sep + "compiler" + os.sep + "tools" + os.sep + "clang"

print
print "**********************************************************************"
print
yesno = raw_input("Do you want to pull and update cppamp-driver [y/n]? ")
if yesno.lower() in [ 'y', 'yes' ]:
  os.chdir(top_dir)
  os.system("hg pull https://bitbucket.org/multicoreware/cppamp-driver")
elif yesno in [ 'q', 'exit' ]:
  sys.exit(0)
else:
  print "skipped."

print
print "**********************************************************************"
print
yesno = raw_input("Do you want to pull and update cppamp [y/n]? ")
if yesno.lower() in [ 'y', 'yes' ]:
  os.chdir(clang_dir)
  os.system("hg pull https://bitbucket.org/multicoreware/cppamp")
elif yesno in [ 'q', 'exit' ]:
  sys.exit(0)
else:
  print "skipped."

print
print "Done! You have to 'hg update' for those updated repositories."
print

