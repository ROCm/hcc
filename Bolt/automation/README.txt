############################################################################
#   Copyright 2012 - 2013 Advanced Micro Devices, Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
############################################################################

This folder contains automatic benchmarking, building, testing and packaging scripts for Bolt.

1. bolt_bench.cmd - Benchmarking script.

   Usage:
   > bolt_bench.cmd [[--bin-path] [path-to-benchmark-binary]] [[--prefix] [benchmark-prefix]]

   Options:
   --bin-path
   Specifies the path of benchmark file.
   e.g. <path-to-benchmark-binary>/clBolt.Bench.Benchmark.exe

   --prefix
   Specifies the name of the output xml file.
   e.g bench.scan.xml

   Default:
   > bolt_bench.cmd
   Where, bin-path is current directory
          prefix is bench


2. bolt_build.cmd - Build script. Also responsible for packaging Bolt.

    Usage:
    > bolt_build.cmd [-h] [[--source] [source-path]] [[--install] [install-path]] [[--os] [os]] 
                     [[--os-version][os-version]] [[--comp] [compiler]] [[--comp-version] [compiler-version]]
                     [[--bit] [bitness]] [[--config] [configuration]] [[--version-major] [version-major]]
                     [[--version-minor] [version-minor]] [[--version-patch] [version-patch]]

    Options:
    -h
    Help!

    --source
    Specifies the path to the repository Bolt/. Default is the path on Jenkins.

    --install
    Specifies the path to the install build. Default is current directory.

    --os
    Specifies the name of the OS. Default is Win.

    --os-version
    Specifies the version of the OS. Default is 7.

    --comp
    Specifies compiler used. Default is VS.

    --comp-version
    Specifies compiler version. Default is 11.

    --bit
    Specifies the bitness of the build. Default is 64.

    --config
    Specifies the build configuration. Default is Release.

    --version-major
    Specifies the version (major) of the release.

    --version-minor
    Specifies the version (minor) of the release.

    --version-patch
    Specifies the version (patch) of the release.

    e.g. > bolt_build.cmd --source Bolt/ --install Build/ --os Win --os-version 8 --comp VS --comp-version 10 ^
                          --bit 32 --config Debug --version-major 1 --version-minor 0 --version-patch 73


    Default:
    > bolt_build.cmd --source <path-to-repository>

    NOTE: AMP build will be disables if VS10 is specifies as the compiler.

3. bolt-test.cmd - Test automation script.

    Usage:
    > bolt_test.cmd [-h] [[--bin-path] [binary-path]] [[--results-path] [results-path]]
                    [[--test-name] [test-output-prefix]] [[--files] [filter-string]]

    Options:
    -h
    Help!

    --bin-path
    Specifies the path to the binary files (executables).

    --results-path
    Specifies the path to the results folder.

    --test-name
    Specifies the test output prefix.

    --files
    Specifies the name of the test. Default is a filter string.

    e.g > bolt_test.cmd --bin-path Build/Bolt-Build/staging/debug --results-path Results/
                        --test-name clBolt.Test --files clBolt.Test.Control

4. filename.append.py - Renames the built package. Called internally by bolt_build.cmd
