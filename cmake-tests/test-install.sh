#!/bin/bash

set -e

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SOURCE_DIR="$( cd -P "$( dirname "$SOURCE" )" && cd .. && pwd )"

TMP_DIR=$(mktemp -d)

INSTALL_DIR=$TMP_DIR/hcc

mkdir $TMP_DIR/hcc-build && cd $TMP_DIR/hcc-build

cmake $SOURCE_DIR -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR $@
make -j`nproc` world
make -j`nproc`
make install

mkdir $TMP_DIR/test-build && cd $TMP_DIR/test-build

CXX=$TMP_DIR/hcc/bin/hcc cmake $SOURCE_DIR/cmake-tests -DCMAKE_PREFIX_PATH=$INSTALL_DIR
make

rm -rf $TMP_DIR
