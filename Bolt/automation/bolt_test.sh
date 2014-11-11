#SetLocal EnableDelayedExpansion
################################################################################################
# Master Bolt Build Script
################################################################################################
HR="###############################################################################"
testStartTime=`date`

################################################################################################
# File Paths
BOLT_BUILD_INSTALL_PATH=/GitRoot/BoltBuilds/gcc64.SuperBuild
BOLT_TEST_BIN_PATH=$BOLT_BUILD_INSTALL_PATH/Bolt-build/Staging/Debug
BOLT_TEST_RESULTS_PATH=`pwd`
BOLT_TEST_BATCH_NAME=Bolt.Test.*.*


################################################################################################
# Default test parameters
BOLT_TEST_FILE_FILTER_STRING=clBolt.Test.*.*

################################################################################################
# Read command line parameters
args=$#
for((i=0;i<$args;i++))
do
  if [ "$1" == "" ]; then 
	break
  fi

  if [ "$1" == "-h" ]; then
	echo "Build script for Bolt"
	echo "Command line options:" 
	echo "-h     ) Print help"
        echo "--bin-path ) Test binaries path"
        echo "--test-name ) Test Name BUID_ID"
	echo "--files  ) Files to test"
	exit	
  fi 	

  if [ "$1" == "--bin-path" ]; then
     BOLT_TEST_BIN_PATH=$2
    shift
  fi

  if [ "$1" == "--results-path" ]; then
    BOLT_TEST_RESULTS_PATH=$2
    shift
   fi

  if [ "$1" == "--test-name" ]; then
    BOLT_TEST_BATCH_NAME=$2
    shift
  fi
  if [ "$1" == "--files" ]; then
    BOLT_TEST_FILE_FILTER_STRING=$2
    shift
   fi

shift
done


################################################################################################
# Print Info
################################################################################################
echo Bin Path:      $BOLT_TEST_BIN_PATH
echo Results Path:  $BOLT_TEST_RESULTS_PATH
echo Batch Name:    $BOLT_TEST_BATCH_NAME
echo Filter String: $BOLT_TEST_FILE_FILTER_STRING


################################################################################################
# Move to Bin Directory and Run Tests
################################################################################################
echo Moving into $BOLT_TEST_BIN_PATH
CURDIR=`pwd`
cd $BOLT_TEST_BIN_PATH
echo Now in  `pwd`
ls -1

for f in $BOLT_TEST_FILE_FILTER_STRING; 
do 
  
  echo . $HR
  echo Testing: $f
  CMDTORUN="--gtest_output=xml:$BOLT_TEST_RESULTS_PATH/$BOLT_TEST_BATCH_NAME_"$f"_.xml > $BOLT_TEST_RESULTS_PATH/$BOLT_TEST_BATCH_NAME_"$f"_.log 2>&1"
echo $CMDTORUN
 ./$f $CMDTORUN
done



################################################################################################
# End
################################################################################################

################################################################################################
echo "Moving back to original dir" $CURDIR
cd $CURDIR
echo .
echo $HR
echo "Done. StartTime={%testStartTime} StopTime={`date`}"
echo $HR

