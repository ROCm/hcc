
// RUN: %hc %s -o %t.out && %t.out

// Check if __int128 could be accepted in kernel compilation path
int main() {
  __int128 v = (__int128)0;
  return (int)v;
}
