// RUN: %not %hc %s -o %t.out 2>&1 | %not grep 'Segmentation fault'

void func(void);

int main() {
  func();
  return 0;
}

