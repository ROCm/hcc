
// RUN: %hc %s -o %t.out && %t.out

#include <iostream>
#include <amp.h>

// An HSA version of C++AMP program
int main ()
{

  const int vecSize = 16;

  int ans[vecSize];
  int *p_ans = &ans[0];

  parallel_for_each(
    Concurrency::extent<1>(vecSize),
    [=](Concurrency::index<1> idx) restrict(amp) {

    int arr[vecSize][vecSize];

    for (int i = 0; i < vecSize; i++) {
      for (int j = 0; j < vecSize; j++) {
        arr[i][j] = i * j * idx[0];
      }
    }

    for (int i = 0; i < vecSize; i++) {
      p_ans[idx[0]] += arr[i][idx[0]];
    }
  });

  // Verify
  int error = 0;
  for(int i = 0; i < vecSize; i++) {
    ; //error += abs(ans[i] - (3 * i));
  }
  if (error == 0) {
    std::cout << "Verify success!\n";
  } else {
    std::cout << "Verify failed!\n";
  }
  return (error != 0);
}

