#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

extern void plot(const std::string &filename, const std::vector<std::chrono::duration<double>> &data);
extern void remove_outliers(std::vector<std::chrono::duration<double>> &data,
                     std::vector<std::chrono::duration<double>> &outliers);
void printVecInfo(const std::string &name, const std::vector<std::chrono::duration<double>> &data);


template <typename T>
T average(const std::vector<std::chrono::duration<T>> &data) {
  T avg_duration = 0;

  for(auto &i : data)
    avg_duration += i.count();

  return avg_duration/data.size();
}
