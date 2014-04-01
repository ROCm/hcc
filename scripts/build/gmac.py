#!/usr/bin/env python

import os
import sys
from mercurial import ui, hg, commands

url = "https://bitbucket.org/multicoreware/gmac"

def download(url, target):
  if not os.path.exists(target):
    commands.clone(ui.ui(), url, target)

def build(target, inc_dir, lib_dir, prefix):
  # platform dependent
  # ./configure
  if sys.platform.find('linux') != -1: # linux
    os.chdir(target)
    os.system("./configure --enable-installer --enable-opencl --with-opencl-include=%s --with-opencl-library=%s --enable-tests --enable-debug --prefix=%s" % (inc_dir, lib_dir, prefix))
    os.system("make")
  else:
    print "Error: not supported platform."
    sys.exit(1)
  pass

def patch(target):
  os.chdir(target)
  print "Patching GMAC"
  os.system("patch -Np1 < %s/../scripts/build/gmac.patch" % target)

def test():
  result = os.system("./oclVecAdd")
  if result != 0:
    print "Error: gmac: test failed."
    sys.exit(1)

def run(url, target, inc_dir, lib_dir, prefix):
  download(url, target)
  patch(target)
  build(target, inc_dir, lib_dir, prefix)
  test()

target = sys.argv[1]
inc_dir = sys.argv[2]
lib_dir = sys.argv[3]
prefix = sys.argv[4]
onoff = sys.argv[5]
if onoff == 'ON':
  run(url, target, inc_dir, lib_dir, prefix)
