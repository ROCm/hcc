// RUN: %hc %s -o %t.out && %t.out

#include <hc.hpp>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

// NOTE: This program only works with lazy initialization
// enabled in the HCC runtime, which defers the initialization
// of the HSA/ROCm runtime until the first of an HC functionalty
// (after fork() in the example).

int do_something_and_verify(const int child) {
  constexpr int grid_size = 2048;
  hc::array_view<int> av(grid_size);
  hc::parallel_for_each(av.get_extent(),
                        [=](hc::index<1> idx) [[hc]] {
                          av[idx] = idx[0] * child;
                        });

  int num_errors = 0;
  for (int i = 0; i < grid_size; i++) {
    if (av[i] != i*child)
      ++num_errors;
  }
  return num_errors;
}

int main() {
  constexpr bool failed = true;
  constexpr int num_child = 4;
  std::vector<pid_t> child_pids;
  for (int i = 0; i < num_child; ++i) {
    pid_t pid = fork();
    if (pid == -1) {
      // something bad with fork(), quit
      return failed;
    }

    if (pid == 0) {
      // child
      return !(do_something_and_verify(i)==0);
    }
    else {
      // parent
      child_pids.push_back(pid);
    }
  }

  for (auto p : child_pids) {
    int status;
    if (waitpid(p, &status, 0) == -1) {
      return failed;
    }
    if (WIFEXITED(status)
        && WEXITSTATUS(status) == 0) {
        continue;
    }
    else {
      // something wrong happen in the child
      return failed;
    }
  }
  return !failed;
}
