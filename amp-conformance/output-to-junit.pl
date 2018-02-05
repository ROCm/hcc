
print "<testsuite>\n";

while ( <STDIN> ) {

  #Line 1:
  #Test: TestName

  $_ =~ /Test: (.+)/ or die "Failed to find testname: $_\n";

  my $parsed_name = $1;

  # Remove test.cpp from end.
  $parsed_name =~ s#/test.cpp##;

  # Remove directory names from front.
  $parsed_name =~ /Tests\/(.+)/;

  # Extract the classname and name
  my $full_name = $1;
  my @name_parts = split('/', $full_name);

  my $name = pop(@name_parts);
  my $classname = join('.', @name_parts);

  # Skip the next line:
  # Compile only:
  <stdin>;

  # Skip the next line:
  # Expected success:
  <stdin>;

  my $result = '';
  my $subtest = -1;
  my $stdout = '';
  while ( <stdin> ) {

    # This marks the beginning of a new sub-test.
    if ($_ =~ /^Command:/) {
      # Emit the previous subtest if any.
      if ($subtest > -1) {
        my $subtest_name = get_subtest_name($name, $subtest);
        print_test_result($classname, $subtest_name, $result, $stdout);
      }

      $subtest++;
      $result ='';
      $stdout = $_;
      next;
    }

    # Check for test result
    # This marks the end of a sub-test.
    if ($_ =~ /Result: (.+)/) {
      if ($1 eq 'skipped') {
        $result = '<skipped/>';
      } elsif ($1 eq 'failed') {
        $result = '<failure/>';
      } elsif ($1 eq 'invalid or cannot open') {
        $result = '<failure/>';
      }
      next;
    }

    # Check for test delimiter.  This marks the end of a test.
    if ($_ =~ /^=+$/) {
      my $subtest_name = $name;
      if ($subtest > 0) {
        $subtest_name = get_subtest_name($name, $subtest);
      }
      print_test_result($classname, $subtest_name, $result, $stdout);
      last;
    }

    # Everything else gets appended to stdout:
    $stdout .= $_;
  }
}

print "</testsuite>\n";

sub print_test_result {
  my ($classname, $name, $result, $stdout) = @_;

  # To keep the output small, we only save stdout for failed tests.
  if ( $result ne '<failure/>') {
    $stdout = '';
  } else {
    # Escape special characters
    $stdout =~ s/&/&amp;/g;
    $stdout =~ s/</&lt;/g;
  }
  print "<testcase classname='$classname' name='$name'><system-out>$stdout</system-out>$result</testcase>\n";
}

sub get_subtest_name {
  my ($basename, $sub_id) = @_;
  return $basename . '.SubTest' . $sub_id;
}
