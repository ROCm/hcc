@echo off
REM SetLocal EnableDelayedExpansion
REM ################################################################################################
REM # Master Bolt Build Script
REM ################################################################################################
set HR=###############################################################################
set testStartTime=%time%

REM ################################################################################################
REM # File Paths
set BOLT_BUILD_INSTALL_PATH=C:\GitRoot\BoltBuilds\VS11Win64.SuperBuild
set BOLT_TEST_BIN_PATH=%BOLT_BUILD_INSTALL_PATH%\Bolt-build\Staging\Debug
set BOLT_TEST_RESULTS_PATH=%CD%
set BOLT_TEST_BATCH_NAME=Bolt.Test


REM ################################################################################################
REM # Default test parameters
set BOLT_TEST_FILE_FILTER_STRING=clBolt.Test.*.exe

REM ################################################################################################
REM # Read command line parameters
:Loop
  IF [%1]==[] GOTO Continue

  if /i "%1"=="-h" (
    goto :print_help
  )
  if /i "%1"=="--bin-path" (
    set BOLT_TEST_BIN_PATH=%2
    SHIFT
  )
  if /i "%1"=="--results-path" (
    set BOLT_TEST_RESULTS_PATH=%2
    SHIFT
  )
  if /i "%1"=="--test-name" (
    set BOLT_TEST_BATCH_NAME=%2
    SHIFT
  )
  if /i "%1"=="--files" (
    echo %2
    set BOLT_TEST_FILE_FILTER_STRING=%2
    SHIFT
  )
SHIFT
GOTO Loop
:Continue


REM ################################################################################################
REM # Print Info
REM ################################################################################################
echo Bin Path:      %BOLT_TEST_BIN_PATH%
echo Results Path:  %BOLT_TEST_RESULTS_PATH%
echo Batch Name:    %BOLT_TEST_BATCH_NAME%
echo Filter String: %BOLT_TEST_FILTER_STRING%


REM ################################################################################################
REM # Move to Bin Directory and Run Tests
REM ################################################################################################
echo Moving into %BOLT_TEST_BIN_PATH%
pushd %BOLT_TEST_BIN_PATH%
echo Now in %CD%
dir

for %%f in (%BOLT_TEST_FILE_FILTER_STRING%) do (
  echo.
  echo %HR%
  echo %%f
  REM set TestFile=%%~nxf
  echo Testing: %%f
  REM set COMMAND_STRING="%%f --gtest_output="xml:%BOLT_TEST_RESULTS_PATH%\%BOLT_TEST_BATCH_NAME%_%TestFile%_.xml"
  REM echo %COMMAND_STRING%
  call %%f --gtest_output="xml:%BOLT_TEST_RESULTS_PATH%\%BOLT_TEST_BATCH_NAME%_%%f_.xml" > %BOLT_TEST_RESULTS_PATH%\%BOLT_TEST_BATCH_NAME%_%%f_.log 2>&1
)


REM ################################################################################################
REM # End
REM ################################################################################################
goto :Done


REM ################################################################################################
REM Print Help
:print_help
echo Build script for Bolt
echo Command line options: 
echo -h     ) Print help
echo -debug ) Create and package a debug build of Amd.clFFT with debug files
echo -Win32 ) Build a 32bit (default: 64bit)
echo -VS10  ) Build with VS10 (default: VS11)
if exist %BOLT_BUILD_PATH%\success del /Q /F %BOLT_BUILD_PATH%\success
goto :Done


REM ################################################################################################
REM Done, Clean up
:Done
popd
echo.
echo %HR%
echo "Done. StartTime={%testStartTime%} StopTime={%time%}"
echo %HR%
goto :eof
