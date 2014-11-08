################################################################################################
# Master Bolt Build Script
################################################################################################
HR='###############################################################################'
buildStartTime=`date`
#save and restore PATH else infinitely lengthened
OLD_SYSTEM_PATH=$PATH
CMAKE=cmake
MAKE_THREADS=4 
################################################################################################
# File Paths

BOLT_BUILD_INSTALL_PATH=`pwd`
BOLT_BUILD_SOURCE_PATH=$(dirname $0)/..

################################################################################################
# Build Version
BOLT_BUILD_VERSION_MAJOR_FILE=$BOLT_BUILD_SOURCE_PATH/automation/bolt.version.major
BOLT_BUILD_VERSION_MINOR_FILE=$BOLT_BUILD_SOURCE_PATH/automation/bolt.version.minor
BOLT_BUILD_VERSION_PATCH_FILE=$BOLT_BUILD_SOURCE_PATH/automation/bolt.version.patch
BOLT_BUILD_VERSION_MAJOR=
BOLT_BUILD_VERSION_MINOR=
BOLT_BUILD_VERSION_PATCH=

################################################################################################
# BUILD TYPES
BUILD_RELEASE=false
BUILD_RELEASE=false
###############################################################################################


if test -s  $BOLT_BUILD_VERSION_MAJOR_FILE
then
BOLT_BUILD_VERSION_MAJOR=`<$BOLT_BUILD_VERSION_MAJOR_FILE`
else
echo $BOLT_BUILD_VERSION_MAJOR_FILE not found.
fi

if test -s  $BOLT_BUILD_VERSION_MINOR_FILE
then 
BOLT_BUILD_VERSION_MINOR=`<$BOLT_BUILD_VERSION_MINOR_FILE`
else
echo $BOLT_BUILD_VERSION_MINOR_FILE not found.
fi

if test -s $BOLT_BUILD_VERSION_PATCH_FILE
then
BOLT_BUILD_VERSION_PATCH=`<$BOLT_BUILD_VERSION_PATCH_FILE`
else
echo $BOLT_BUILD_VERSION_PATCH_FILE not found.
fi

################################################################################################
# Default build parameters
BOLT_BUILD_OS=Linux
BOLT_BUILD_OS_VER=
BOLT_BUILD_COMP=gcc
BOLT_BUILD_COMP_VER=4.6.3
BOLT_BUILD_BIT=64
BOLT_BUILD_USE_AMP=OFF
BOLT_VERSION=$BOLT_BUILD_VERSION_MAJOR.$BOLT_BUILD_VERSION_MINOR.$BOLT_BUILD_VERSION_PATCH

################################################################################################
# Read command line parameters
args=$#
for((i=0;i<$args;i++))
do
  if [ "$1" == "" ]; then
    break
  fi
    
  if [ "$1" == "-h" ]; then
    ################################################################################################
    # Help
    echo "Build script for Bolt"
    echo "Command line options:" 
    echo "-h    ) Print help"   
    echo "--bit 32 ) Build a 32bit (default: 64bit)"
    echo "--release ) Build release Builds"
    echo "--debug   ) Build debug   Builds"

    if test -s  $BOLT_BUILD_PATH/success 
        then 
     rm -rf %BOLT_BUILD_PATH%\success
    fi
    exit    
  fi
  
  if [ "$1" == "--source" ]; then
   BOLT_BUILD_SOURCE_PATH=$2
  shift 
  fi    
  
  if [ "$1" == "--install" ]; then
  BOLT_BUILD_INSTALL_PATH=$2
  shift
  fi

  if [ "$1" == "--os" ]; then 
   BOLT_BUILD_OS=$2
   shift
  fi

  if [ "$1" == "--os-ver" ]; then
   BOLT_BUILD_OS_VER=$2
   shift
  fi

  if [ "$1" == "--comp" ]; then
     BOLT_BUILD_COMP=$2
     shift
  fi
  
  if [ "$1" == "--comp-ver" ]; then
   BOLT_BUILD_COMP_VER=$2
   shift
   fi

  if [ "$1" == "--bit" ]; then 
   BOLT_BUILD_BIT=$2
   shift
  fi
  
  if [ "$1" == "--version-major" ]; then
    BOLT_BUILD_VERSION_MAJOR=$2
    shift
  fi

  if [ "$1" == "--version-minor" ]; then
   BOLT_BUILD_VERSION_MINOR=$2
  shift 
  fi
  
  if [ "$1" == "--version-patch" ]; then
   BOLT_BUILD_VERSION_PATCH=$2
  shift
  fi

  if [ "$1" == "--mt" ]; then
   MAKE_THREADS=$2
  shift
  fi

  if [ "$1" == "--release" ]; then
   BUILD_RELEASE=true
  fi

  if [ "$1" == "--debug" ]; then
   BUILD_DEBUG=true
  fi

shift
done



################################################################################################
# Construct Build Parameters

BOLT_BUILD_CMAKE_GEN="Unix Makefiles"

if [ "$BOLT_BUILD_BIT" == "64" ]; then
 BOLT_X64=ON
 else 
  BOLT_X64=OFF  
fi
#REM translate versions to command line flags
BOLT_BUILD_FLAG_MAJOR=
BOLT_BUILD_FLAG_MINOR=
BOLT_BUILD_FLAG_PATCH=

if [ "$BOLT_BUILD_VERSION_MAJOR" != "" ]; then
  BOLT_BUILD_FLAG_MAJOR="-D Bolt.SuperBuild_VERSION_MAJOR=$BOLT_BUILD_VERSION_MAJOR"
fi
if [ "$BOLT_BUILD_VERSION_MINOR" != "" ]; then
  BOLT_BUILD_FLAG_MINOR="-D Bolt.SuperBuild_VERSION_MINOR=$BOLT_BUILD_VERSION_MINOR"
fi
if [ "$BOLT_BUILD_VERSION_PATCH" != "" ]; then
 BOLT_BUILD_FLAG_PATCH="-D Bolt.SuperBuild_VERSION_PATCH=$BOLT_BUILD_VERSION_PATCH"
fi


#REM ################################################################################################
#REM # Print Build Info
echo .
echo $HR
echo Info: Bolt Build Parameters
echo Info: Source:    $BOLT_BUILD_SOURCE_PATH
echo Info: Install:   $BOLT_BUILD_INSTALL_PATH
echo Info: OS:        $BOLT_BUILD_OS $BOLT_BUILD_OS_VER$
echo Info: Compiler:  $BOLT_BUILD_COMP $BOLT_BUILD_COMP_VER $BOLT_BUILD_BIT $bit
echo Info: CMake Gen: $BOLT_BUILD_CMAKE_GEN
echo Info: Major:     $BOLT_BUILD_FLAG_MAJOR
echo Info: Minor:     $BOLT_BUILD_FLAG_MINOR
echo Info: Patch:     $BOLT_BUILD_FLAG_PATCH
echo Building with : $MAKE_THREADS threads


################################################################################################
#REM Echo a blank line into a file called success; the existence of success determines whether we built successfully


#REM Specify the location of a local image of boost, Google test and doxygen. 
#REM Currently BOLT uses Boost 1.52.0, Doxygen 1.8.3.windows, Google Test 1.6.0 versions
#REM and TBB version 4.1 update 2. 
#REM set BOOST_URL=<Enter path to Boost folder>/boost_1_52_0.zip
#REM set DOXYGEN_URL=<Enter path to Doxygen zip file>/doxygen-1.8.3.windows.bin.zip
#REM set GTEST_URL=<Enter path to GTEST folder>/gtest-1.6.0.zip
#REM set TBB_ROOT=<Enter path to TBB folder>

#REM Otherwise The above 4 variables can also be defined in the environment variable. 

################################################################################################
# Start of build logic here
################################################################################################

################################################################################################
# Cmake

if [ "$BUILD_RELEASE" = "true" ]; then


echo . > $BOLT_BUILD_INSTALL_PATH/success

mkdir release
cd release
echo .
echo $HR
echo Info: Running CMake to generate build files.
$CMAKE\
  -G "$BOLT_BUILD_CMAKE_GEN"\
  -D BUILD_AMP=$BOLT_BUILD_USE_AMP\
  -D BUILD_StripSymbols=ON\
  -D BUILD_TBB=ON\
  -D Bolt_BUILD64=$BOLT_X64\
  -D BOLT_BUILD_TYPE=RELEASE\
  -D Bolt.SuperBuild_VERSION_PATCH=$BOLT_BUILD_VERSION_PATCH\
  -D Bolt.SuperBuild_VERSION_MAJOR=$BOLT_BUILD_VERSION_MAJOR\
  -D Bolt.SuperBuild_VERSION_MINOR=$BOLT_BUILD_VERSION_MINOR\
  $BOLT_BUILD_SOURCE_PATH/superbuild
  
if [ "$?" != "0" ]; then
  echo Info: CMake failed.
  rm -rf $BOLT_BUILD_INSTALL_PATH/success
  exit 
fi
cd ..

################################################################################################
# Super Build -- Release
echo .
echo $HR
echo Info: Running Make for Release SuperBuild.
cd release
make -j$MAKE_THREADS 
cd Bolt-build/doxy
make Bolt.Documentation
cd ..
make package


#REM Rename the package that we just built
#REM I do this here because I can not figure out how to get cpack to append the configuration string
echo "python $BOLT_BUILD_SOURCE_PATH/automation/filename.append.py *.tar.gz -release -debug -release"
python $BOLT_BUILD_SOURCE_PATH/automation/filename.append.py *.tar.gz -release -debug -release

cd ../..

fi
################################################################################################
# Cmake


if [ "$BUILD_DEBUG" = "true" ]; then

echo . > $BOLT_BUILD_INSTALL_PATH/success

mkdir debug
cd debug
echo .
echo $HR
echo Info: Running CMake to generate build files.
$CMAKE\
  -G "$BOLT_BUILD_CMAKE_GEN"\
  -D BUILD_AMP=$BOLT_BUILD_USE_AMP\
  -D BUILD_StripSymbols=ON\
  -D BUILD_TBB=ON\
  -D Bolt_BUILD64=$BOLT_X64\
  -D BOLT_BUILD_TYPE=DEBUG\
  -D Bolt.SuperBuild_VERSION_PATCH=$BOLT_BUILD_VERSION_PATCH\
  -D Bolt.SuperBuild_VERSION_MAJOR=$BOLT_BUILD_VERSION_MAJOR\
  -D Bolt.SuperBuild_VERSION_MINOR=$BOLT_BUILD_VERSION_MINOR\
  $BOLT_BUILD_SOURCE_PATH/superbuild

if [ "$?" != "0" ]; then
  echo Info: CMake failed.
  rm -rf $BOLT_BUILD_INSTALL_PATH/success
  exit
fi
cd ..

################################################################################################
# Super Build -- Debug
echo .
echo $HR
echo Info: Running Make for Debug SuperBuild

cd debug
make -j$MAKE_THREADS 
cd Bolt-build/doxy
make Bolt.Documentation
cd ..
make package

#REM Rename the package that we just built
#REM I do this here because I can not figure out how to get cpack to append the configuration string
echo "python $BOLT_BUILD_SOURCE_PATH/automation/filename.append.py *.tar.gz -debug -debug -release"
python $BOLT_BUILD_SOURCE_PATH/automation/filename.append.py *.tar.gz -debug -debug -release

cd ../..

fi

echo $HR
endtime=`date`
echo "Done. StartTime={$buildStartTime} StopTime={$endtime}"
echo $HR


