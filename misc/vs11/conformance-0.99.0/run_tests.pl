#!/bin/perl
##############################################################################################
# Copyright (c) Microsoft
#
# All rights reserved
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file 
# except in compliance with the License. You may obtain a copy of the License at 
# http://www.apache.org/licenses/LICENSE-2.0  
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER 
# EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OR CONDITIONS OF 
# TITLE, FITNESS FOR A PARTICULAR PURPOSE, MERCHANTABLITY OR NON-INFRINGEMENT. 
#
# See the Apache Version 2.0 License for specific language governing permissions and 
# limitations under the License.
##############################################################################################
use Cwd;
use File::Find;
use Safe;
use strict;

### Compiler configuration
my $cflag_compile_only = '/c';
my $cflag_define = '/D%s#"%s"'; # to be used as sprintf($cflag_define, "NAME", "VALUE");
my $test_exec = 'test.exe';

### Environment configuration
my $tests_root = './Tests';
my $run_log = 'run.log';
my $cc = $ENV{"CC"};
my $cflags = $ENV{"CFLAGS"};

### Prepare environment
if(-e $run_log)
{
	exit_message(1, "Log file '$run_log' already exists!");
}
if($cc eq "")
{
	exit_message(1, "No compiler defined!");
}

### Find tests
my @tests;
sub match_tests
{	
	if(lc($_) eq 'test.cpp')
	{
		push @tests, cwd().'/'.$_;
	}
}
find(\&match_tests, $tests_root);

### Execute tests
use constant PASS => 0;
use constant SKIP => 1;
use constant FAIL => 2;
my $num_passed = 0;
my $num_skipped = 0;
my $num_failed = 0;

foreach my $test (@tests)
{
	log_message("Test: $test");
	
	# Read test configuration
	undef %Test::config;
	my $conf_file = (substr lc($test), 0, -3).'conf';
	if(-e $conf_file)
	{
		my $safe = new Safe('Test');
		$safe->rdo($conf_file) or &exit_message(1, "Cannot open $conf_file");
	}
	
	if(not defined $Test::config{'definitions'})
	{
		$Test::config{'definitions'} = [{}];
	}
	
	# Find "expects error" directives in cpp
	open(TEST_CPP, $test) or &exit_message(1, "Cannot open $test");
	$Test::config{'expected_success'} = (grep m@//#\s*Expects\s*(\d*)\s*:\s*(warning|error)@i, <TEST_CPP>) == 0;
	close(TEST_CPP);
	
	log_message('Compile only: '.bool_str($Test::config{'compile_only'})."\n"
		.'Expected success: '.bool_str($Test::config{'expected_success'}));
	
	# For each set of definitions
	foreach my $def_set (@{$Test::config{'definitions'}})
	{
		print "$test : ";
	
		# Build and execute test
		my $cflags_defs = '';
		while(my ($k, $v) = each(%{$def_set}))
		{
			$cflags_defs = $cflags_defs.sprintf($cflag_define.' ', $k, $v);
		}
		my $command = "$cc $cflags $cflags_defs "
			.($Test::config{'compile_only'} ? $cflag_compile_only : "")
			." $test 2>>$run_log 1>>&2";
		
		log_message("Command: $command\n"
			."Build output:\n"
			."<<<");
		my $build_exit_code = system($command) >> 8;
		log_message(">>>\n"
			."Build exit code: $build_exit_code");
	
		my $exec_exit_code;
		if((not $Test::config{'compile_only'}) && $build_exit_code == 0 && $Test::config{'expected_success'})
		{
			log_message("Execution output:\n"
				.'<<<');
			$exec_exit_code = system("$test_exec 2>>$run_log 1>>&2") >> 8;
			log_message(">>>\n"
				."Execution exit code: $exec_exit_code");
		}
	
		# Interpret result
		my $result;
		if(not $Test::config{'expected_success'}) # Negative test
		{
			if($build_exit_code != 0)
			{
				$result = PASS;
			}
			else
			{
				$result = FAIL;
			}
		}
		elsif($Test::config{'compile_only'}) # Compile only test
		{
			if($build_exit_code == 0)
			{
				$result = PASS;
			}
			else
			{
				$result = FAIL;
			}
		}
		else # Executable test
		{
			if($build_exit_code != 0)
			{
				$result = FAIL;
			}
			elsif($exec_exit_code == 0)
			{
				$result = PASS;
			}
			elsif($exec_exit_code == 2)
			{
				$result = SKIP;
			}
			else
			{
				$result = FAIL;
			}
		}
		
		if($result == PASS)
		{
			$num_passed++;
			print "passed\n";
			log_message('Result: passed');
		}
		elsif($result == FAIL)
		{
			$num_failed++;
			print "failed\n";
			log_message('Result: failed');
		}
		elsif($result == SKIP)
		{
			$num_skipped++;
			print "skipped\n";
			log_message('Result: skipped');
		}
		else
		{
			exit_message(1, "Unexpected result!");
		}
	}
	log_message("=====================================================");
}

### Print summary
my $num_total = $num_passed + $num_skipped + $num_failed;
print "==========================\n";
if($num_total != 0)
{
	printf(" Passed:  %d (%.3f%%)\n", $num_passed,  $num_passed / $num_total * 100);
	printf(" Skipped: %d (%.3f%%)\n", $num_skipped, $num_skipped / $num_total * 100);
	printf(" Failed:  %d (%.3f%%)\n", $num_failed,  $num_failed / $num_total * 100);
}
print " Total:  $num_total\n";
print "==========================\n";

### Subroutines
# Use: exit_message(code, msg)
sub exit_message
{
	if(@_ != 2) { die('exit_message expects 2 arguments'); }
	print("\n".($_[0] == 0 ? 'SUCCESS' : 'FAILURE').": ".$_[1]);
	exit($_[0]);
}

# Use: log_message(msg, ...)
sub log_message
{
	open(FH, ">>", $run_log) or &exit_message(1, "Cannot open $run_log");
	print FH "@_\n";
	close(FH);
}

# Use: bool_str(val)
# Returns: string 'true'/'false'
sub bool_str
{
	return $_[0] ? 'true' : 'false';
}