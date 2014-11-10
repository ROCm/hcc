@echo off
REM ################################################################################################
REM # Master Bolt Build Script
REM ################################################################################################
set HR=###############################################################################
set buildStartTime=%time%
REM save and restore PATH else infinitely lengthened by vcvarsall
set OLD_SYSTEM_PATH=%PATH%
set CMAKE="C:\Program Files (x86)\CMake 2.8\bin\cmake.exe"

REM ################################################################################################
REM # File Paths
set BOLT_BUILD_INSTALL_PATH=%CD%
set BOLT_BUILD_SOURCE_PATH=%~dp0..

REM ################################################################################################
REM # Build Version
set BOLT_BUILD_VERSION_MAJOR_FILE=%BOLT_BUILD_SOURCE_PATH%\automation\bolt.version.major
set BOLT_BUILD_VERSION_MINOR_FILE=%BOLT_BUILD_SOURCE_PATH%\automation\bolt.version.minor
set BOLT_BUILD_VERSION_PATCH_FILE=%BOLT_BUILD_SOURCE_PATH%\automation\bolt.version.patch
set BOLT_BUILD_VERSION_MAJOR=
set BOLT_BUILD_VERSION_MINOR=
set BOLT_BUILD_VERSION_PATCH=
if exist %BOLT_BUILD_VERSION_MAJOR_FILE% (
  set /p BOLT_BUILD_VERSION_MAJOR=<%BOLT_BUILD_VERSION_MAJOR_FILE%
) else (
echo %BOLT_BUILD_VERSION_MAJOR_FILE% not found.
)
if exist %BOLT_BUILD_VERSION_MINOR_FILE% (
  set /p BOLT_BUILD_VERSION_MINOR=<%BOLT_BUILD_VERSION_MINOR_FILE%
) else (
echo %BOLT_BUILD_VERSION_MINOR_FILE% not found.
)
if exist %BOLT_BUILD_VERSION_PATCH_FILE% (
  set /p BOLT_BUILD_VERSION_PATCH=<%BOLT_BUILD_VERSION_PATCH_FILE%
) else (
echo %BOLT_BUILD_VERSION_PATCH_FILE% not found.
)

REM ################################################################################################
REM # Default build parameters
set BOLT_BUILD_OS=Win
set BOLT_BUILD_OS_VER=7
set BOLT_BUILD_COMP=VS
set BOLT_BUILD_COMP_VER=11
set BOLT_BUILD_BIT=64
set BOLT_BUILD_USE_AMP=ON
set BOLT_CONFIGURATION=Release
set BOLT_VERSION=%BOLT_BUILD_VERSION_MAJOR%.%BOLT_BUILD_VERSION_MINOR%.%BOLT_BUILD_VERSION_PATCH%


REM ################################################################################################
REM # Read command line parameters
:Loop
  IF [%1]==[] GOTO Continue

  if /i "%1"=="-h" (
    goto :print_help
  )
  if /i "%1"=="--source" (
    set BOLT_BUILD_SOURCE_PATH=%2
    SHIFT
  )
  if /i "%1"=="--install" (
    set BOLT_BUILD_INSTALL_PATH=%2
    SHIFT
  )
  if /i "%1"=="--os" (
    set BOLT_BUILD_OS=%2
    SHIFT
  )
  if /i "%1"=="--os-ver" (
    set BOLT_BUILD_OS_VER=%2
    SHIFT
  )
  if /i "%1"=="--comp" (
    set BOLT_BUILD_COMP=%2
    SHIFT
  )
  if /i "%1"=="--comp-ver" (
    set BOLT_BUILD_COMP_VER=%2
    SHIFT
  )
  if /i "%1"=="--bit" (
    set BOLT_BUILD_BIT=%2
    SHIFT
  )
  REM if /i "%1"=="--config" (
    REM set BOLT_CONFIGURATION=%2
    REM SHIFT
  REM )
  if /i "%1"=="--version-major" (
    set BOLT_BUILD_VERSION_MAJOR=%2
    SHIFT
  )
  if /i "%1"=="--version-minor" (
    set BOLT_BUILD_VERSION_MINOR=%2
    SHIFT
  )
  if /i "%1"=="--version-patch" (
    set BOLT_BUILD_VERSION_PATCH=%2
    SHIFT
  )
SHIFT
GOTO Loop
:Continue


REM ################################################################################################
REM # Construct Build Parameters
set BOLT_BUILD_MSBUILD_PLATFORM_TOOLSET=v%BOLT_BUILD_COMP_VER%0
set BOLT_BUILD_CMAKE_GEN=Visual Studio %BOLT_BUILD_COMP_VER%
if "%BOLT_BUILD_BIT%" == "64" (
  set BOLT_BUILD_MSBUILD_PLATFORM=x64
  set BOLT_BUILD_CMAKE_GEN="%BOLT_BUILD_CMAKE_GEN% %BOLT_BUILD_OS%%BOLT_BUILD_BIT%"
) else (
  set BOLT_BUILD_MSBUILD_PLATFORM=x86
  set BOLT_BUILD_CMAKE_GEN="%BOLT_BUILD_CMAKE_GEN%"
)
REM translate versions to command line flags
set BOLT_BUILD_FLAG_MAJOR=
set BOLT_BUILD_FLAG_MINOR=
set BOLT_BUILD_FLAG_PATCH=
if not "%BOLT_BUILD_VERSION_MAJOR%" == "" (
  set BOLT_BUILD_FLAG_MAJOR=-D Bolt.SuperBuild_VERSION_MAJOR=%BOLT_BUILD_VERSION_MAJOR%
)
if not "%BOLT_BUILD_VERSION_MINOR%" == "" (
  set BOLT_BUILD_FLAG_MINOR=-D Bolt.SuperBuild_VERSION_MINOR=%BOLT_BUILD_VERSION_MINOR%
)
if not "%BOLT_BUILD_VERSION_PATCH%" == "" (
  set BOLT_BUILD_FLAG_PATCH=-D Bolt.SuperBuild_VERSION_PATCH=%BOLT_BUILD_VERSION_PATCH%
)


REM ################################################################################################
REM # Print Build Info
echo.
echo %HR%
echo Info: Bolt Build Parameters
echo Info: Source:    %BOLT_BUILD_SOURCE_PATH%
echo Info: Install:   %BOLT_BUILD_INSTALL_PATH%
echo Info: OS:        %BOLT_BUILD_OS%%BOLT_BUILD_OS_VER%
echo Info: Compiler:  %BOLT_BUILD_COMP%%BOLT_BUILD_COMP_VER% %BOLT_BUILD_BIT%bit
echo Info: CMake Gen: %BOLT_BUILD_CMAKE_GEN%
REM echo Info: CMake Config: %BOLT_CONFIGURATION%
echo Info: Platform:  %BOLT_BUILD_MSBUILD_PLATFORM%
echo Info: Toolset:   %BOLT_BUILD_MSBUILD_PLATFORM_TOOLSET%
echo Info: Major:     %BOLT_BUILD_FLAG_MAJOR%
echo Info: Minor:     %BOLT_BUILD_FLAG_MINOR%
echo Info: Patch:     %BOLT_BUILD_FLAG_PATCH%


REM ################################################################################################
REM # Load compiler environment
if "%BOLT_BUILD_COMP_VER%" == "10" ( 
  if not "%VS100COMNTOOLS%" == "" (
    set VCVARSALL="%VS100COMNTOOLS%..\..\VC\vcvarsall.bat"
    set BOLT_BUILD_USE_AMP=OFF
  ) else (
    goto :error_no_VSCOMNTOOLS
  )
) else if "%BOLT_BUILD_COMP_VER%" == "11" ( 
    if not "%VS110COMNTOOLS%" == "" ( 
      set VCVARSALL="%VS110COMNTOOLS%..\..\VC\vcvarsall.bat"
    ) else (
      goto :error_no_VSCOMNTOOLS
    )
) else if "%BOLT_BUILD_COMP_VER%" == "12" ( 
		if not "%VS120COMNTOOLS%" == "" ( 
			set VCVARSALL="%VS120COMNTOOLS%..\..\VC\vcvarsall.bat"
		) else (
			goto :error_no_VSCOMNTOOLS
		)
	) else (
		echo Unrecognized BOLT_BUILD_COMP_VER=%BOLT_BUILD_COMP_VER%
  	)




if "%BOLT_BUILD_BIT%" == "64" ( 
  echo Info: vcvarsall.bat: %VCVARSALL% x86_amd64
  call %VCVARSALL% x86_amd64
)
if "%BOLT_BUILD_BIT%" == "32" (
  echo Info: vcvarsall.bat: %VCVARSALL% x86
  call %VCVARSALL% x86
)
echo Info: Done setting up compiler environment variables.

REM Echo a blank line into a file called success; the existence of success determines whether we built successfully
echo. > %BOLT_BUILD_INSTALL_PATH%\success

REM Specify the location of a local image of boost, Google test and doxygen. 
REM Currently BOLT uses Boost 1.52.0, Doxygen 1.8.3.windows, Google Test 1.6.0 versions
REM and TBB version 4.1 update 2. 
REM set BOOST_URL=<Enter path to Boost folder>/boost_1_52_0.zip
REM set DOXYGEN_URL=<Enter path to Doxygen zip file>/doxygen-1.8.3.windows.bin.zip
REM set GTEST_URL=<Enter path to GTEST folder>/gtest-1.6.0.zip
REM set TBB_ROOT=<Enter path to TBB folder>

REM Otherwise The above 4 variables can also be defined in the environment variable. 

REM ################################################################################################
REM # Start of build logic here
REM ################################################################################################

REM ################################################################################################
REM # Cmake
echo.
echo %HR%
echo Info: Running CMake to generate build files.
%CMAKE% ^
  -G %BOLT_BUILD_CMAKE_GEN% ^
  -D BUILD_AMP=%BOLT_BUILD_USE_AMP% ^
  -D BUILD_StripSymbols=ON ^
  -D BUILD_TBB=ON ^
  -D Bolt.SuperBuild_VERSION_PATCH=%BOLT_BUILD_VERSION_PATCH% ^
  -D Bolt.SuperBuild_VERSION_MAJOR=%BOLT_BUILD_VERSION_MAJOR% ^
  -D Bolt.SuperBuild_VERSION_MINOR=%BOLT_BUILD_VERSION_MINOR% ^
  %BOLT_BUILD_SOURCE_PATH%\superbuild
if errorlevel 1 (
  echo Info: CMake failed.
  del /Q /F %BOLT_BUILD_INSTALL_PATH%\success
  popd
  goto :Done
)

REM ################################################################################################
REM # Super Build -- Debug
echo.
echo %HR%
echo Info: Running MSBuild for Debug SuperBuild
MSBuild.exe ^
  Bolt.SuperBuild.sln ^
  /m ^
  /fl ^
  /flp1:logfile=DebugErrors.log;errorsonly ^
  /flp2:logfile=DebugWarnings.log;warningsonly ^
  /flp3:logfile=DebugBuild.log ^
  /p:Configuration=Debug ^
  /p:PlatformTarget=%BOLT_BUILD_MSBUILD_PLATFORM% ^
  /p:PlatformToolset=%BOLT_BUILD_MSBUILD_PLATFORM_TOOLSET% ^
  /t:build
REM The following error level does not seem to work!  What do the return codes of msbuild mean?
REM if errorlevel 1 (
  REM echo Info: MSBuild failed for SuperBuild.
  REM del /Q /F %BOLT_BUILD_INSTALL_PATH%\success
  REM goto :Done
REM )

REM ################################################################################################
REM # Super Build -- Release
echo.
echo %HR%
echo Info: Running MSBuild for Release SuperBuild.
MSBuild.exe ^
  Bolt.vcxproj ^
  /m ^
  /fl ^
  /flp1:logfile=ReleaseErrors.log;errorsonly ^
  /flp2:logfile=ReleaseWarnings.log;warningsonly ^
  /flp3:logfile=ReleaseBuild.log ^
  /p:Configuration=Release ^
  /p:BuildProjectReferences=false ^
  /p:PlatformTarget=%BOLT_BUILD_MSBUILD_PLATFORM% ^
  /p:PlatformToolset=%BOLT_BUILD_MSBUILD_PLATFORM_TOOLSET% ^
  /t:build
REM The following error level does not seem to work!  What do the return codes of msbuild mean?
REM if errorlevel 1 (
  REM echo Info: MSBuild failed for SuperBuild.
  REM del /Q /F %BOLT_BUILD_INSTALL_PATH%\success
  REM goto :Done
REM )

REM ################################################################################################
REM # Build Documentation - Independent of configuration
echo.
echo %HR%
echo Info: Running MSBuild for Bolt documentation.
pushd Bolt-build
pushd doxy
MSBuild.exe ^
  Bolt.Documentation.vcxproj ^
  /m ^
  /fl ^
  /flp1:logfile=ReleaseErrors.log;errorsonly ^
  /flp2:logfile=ReleaseWarnings.log;warningsonly ^
  /flp3:logfile=ReleaseBuild.log ^
  /p:Configuration=Release ^
  /p:PlatformTarget=%BOLT_BUILD_MSBUILD_PLATFORM% ^
  /p:PlatformToolset=%BOLT_BUILD_MSBUILD_PLATFORM_TOOLSET% ^
  /t:build
if errorlevel 1 (
  echo Info: MSBuild failed for Bolt documentation.
  del /Q /F %BOLT_BUILD_INSTALL_PATH%\success
  popd
  goto :Done
)
popd

REM Inside of the bolt-build directory
REM ################################################################################################
REM # Zip Package - Debug
echo.
echo %HR%
echo Info: Running MSBuild for packaging
MSBuild.exe ^
  PACKAGE.vcxproj ^
  /m ^
  /fl ^
  /flp1:logfile=DebugErrors.log;errorsonly ^
  /flp2:logfile=DebugWarnings.log;warningsonly ^
  /flp3:logfile=DebugBuild.log ^
  /p:Configuration=Debug ^
  /p:PlatformTarget=%BOLT_BUILD_MSBUILD_PLATFORM% ^
  /p:PlatformToolset=%BOLT_BUILD_MSBUILD_PLATFORM_TOOLSET% ^
  /t:build
if errorlevel 1 (
  echo Info: MSBuild failed for Bolt-build.
  del /Q /F %BOLT_BUILD_INSTALL_PATH%\success
  goto :Done
)

REM Rename the package that we just built
REM I do this here because I can not figure out how to get cpack to append the configuration string
echo python %BOLT_BUILD_SOURCE_PATH%\automation\filename.append.py *.zip -debug -debug -release
python %BOLT_BUILD_SOURCE_PATH%\automation\filename.append.py *.zip -debug -debug -release

REM ################################################################################################
REM # Zip Package - Release
echo.
echo %HR%
echo Info: Running MSBuild for Bolt-build.
MSBuild.exe ^
  PACKAGE.vcxproj ^
  /m ^
  /fl ^
  /flp1:logfile=ReleaseErrors.log;errorsonly ^
  /flp2:logfile=ReleaseWarnings.log;warningsonly ^
  /flp3:logfile=ReleaseBuild.log ^
  /p:Configuration=Release ^
  /p:PlatformTarget=%BOLT_BUILD_MSBUILD_PLATFORM% ^
  /p:PlatformToolset=%BOLT_BUILD_MSBUILD_PLATFORM_TOOLSET% ^
  /t:build
if errorlevel 1 (
  echo Info: MSBuild failed for Bolt-build.
  del /Q /F %BOLT_BUILD_INSTALL_PATH%\success
  goto :Done
)

REM Rename the package that we just built
REM I do this here because I can not figure out how to get cpack to append the configuration string
echo python %BOLT_BUILD_SOURCE_PATH%\automation\filename.append.py *.zip -release -debug -release
python %BOLT_BUILD_SOURCE_PATH%\automation\filename.append.py *.zip -release -debug -release
popd

REM ################################################################################################
REM # End
REM ################################################################################################
goto :Done


REM ################################################################################################
REM Cannot find compiler
:error_no_VSCOMNTOOLS
echo ERROR: Cannot determine the location of the VS Common Tools folder. (%VS100COMNTOOLS% or %VS110COMNTOOLS%)
if exist %BOLT_BUILD_PATH%\success del /Q /F %BOLT_BUILD_PATH%\success
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
set PATH=%OLD_SYSTEM_PATH%
echo.
echo %HR%
echo "Done. StartTime={%buildStartTime%} StopTime={%time%}"
echo %HR%
goto :eof
