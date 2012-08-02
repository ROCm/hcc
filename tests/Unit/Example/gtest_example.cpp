// RUN: %gtest_amp %s -o %t1 && %t1
//
// What's in the comment above indicates it will build this file using
// -std=c++amp and all other necessary flags to build. Then the system will 
// run the built program and check its results with all google test cases.

#include <gtest/gtest.h>    // must.

TEST(ExampleTest, Category1) {
  EXPECT_LT(1, 2);
}

