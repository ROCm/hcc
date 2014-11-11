@echo off

REM =============================================
REM	= Bolt Benchmarking
REM = A good explanation of all DOS .cmd commands is available at http://ss64.com/nt/
REM =============================================

set BOLT_BIN_PATH=.
set BOLT_PREFIX=bench
set LENGTH=16777216
set ITERATIONS=100
set RUNMODE=3
set DEVICE=0
set HOSTMEMORY=0

:Loop
  IF [%1]==[] GOTO Continue
	if /i "%1"=="--bin-path" (
    echo %2
		set BOLT_BIN_PATH=%2
    SHIFT
	)
	if /i "%1"=="--prefix" (
		set BOLT_PREFIX=%2
    SHIFT
	)
SHIFT
GOTO Loop
:Continue

%BOLT_BIN_PATH%/clBolt.Bench.Benchmark.exe ^
  --iterations=%ITERATIONS% ^
  --routine=0 ^
  --filename=%BOLT_PREFIX%.scan.xml
%BOLT_BIN_PATH%/clBolt.Bench.Benchmark.exe ^
  --iterations=%ITERATIONS% ^
  --routine=1 ^
  --filename=%BOLT_PREFIX%.transform_scan.xml
%BOLT_BIN_PATH%/clBolt.Bench.Benchmark.exe ^
  --iterations=%ITERATIONS% ^
  --routine=2 ^
  --filename=%BOLT_PREFIX%.scan_by_key.xml
