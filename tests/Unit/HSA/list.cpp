// XFAIL: Linux
// RUN: %cxxamp -Xclang -fhsa-ext %s -o %t.out && %t.out
#include <vector>
#include <iostream>
#include <amp.h>

using namespace concurrency;

class List {
public:
  int data;
  List* next;
};


int main() {

  std::vector<List> nodes(10);
  for(int i = 0; i < nodes.size(); i++) {
    nodes[i].data = i;
    nodes[i].next = &nodes[(i+1)%nodes.size()];
  }
  
  List* head = nodes.data();
  int sum_gpu = 0;
  int sum_cpu = 0;
  int n = nodes.size();

  // test on GPU
  parallel_for_each(extent<1>(1),[=, &sum_gpu](index<1> idx) restrict(amp) {
    List* l = head;
    for (int i = 0; i < n; ++i) {
      sum_gpu += l->data;
      l = l->next;
    }
  });

  // test on CPU
  {
    List* l = head;
    for (int i = 0; i < n; ++i) {
      sum_cpu += l->data;
      l = l->next;
    }
  }

  // verify
  int error = sum_cpu - sum_gpu;
  if (error == 0) {
    std::cout << "Verify success!\n";
  } else {
    std::cout << "Verify failed!\n";
  }

  return error != 0;
}
