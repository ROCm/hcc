#include "statutils.h"
#include <algorithm>

void plot(const std::string &filename, const std::vector<std::chrono::duration<double>> &data) {
  std::ofstream file(filename + ".dat", std::ios_base::out | std::ios_base::trunc);
  file << "#x y\n";
  for(auto i = data.begin(); i != data.end(); i++)
    file << i - data.begin() << ' ' << i->count() << '\n';
  file << "A_mean = " << average(data) << "\n";
  file.close();
}

void remove_outliers(std::vector<std::chrono::duration<double>> &data,
                     std::vector<std::chrono::duration<double>> &outliers) {

  auto tdata = data;
  std::sort(tdata.begin(), tdata.end());

  const int size = tdata.size();
  const bool parity = size % 2;
  /*
         ---------.---------
    |----|   .    |    .   | ----|
         ---------'---------
         Q1       Q2       Q3

    Q2: median
    Q1: first quartile, median of the lower half, ~25% lie below Q1, ~75% lie above Q1
    Q3: third quartile, median of the upper half ~ 75% lie below Q3, ~25% lie above Q3
    IQR: interquartile range, distance between Q3 and Q1
    outliers: any value below Q1 - 1.5*IQR or above Q3 + 1.5*IQR
  */

  const double Q2 = parity ? tdata[size/2].count() : (tdata[size/2].count() + tdata[size/2 - 1].count())/2;
  const double Q1 = (tdata[size/4].count() + tdata[size/4 - 1].count())/2;
  const double Q3 = (tdata[size - size/4].count() + tdata[size - size/4 - 1].count())/2;

  const double IQR = Q3 - Q1;
  const double lwrB = Q1 - 1.5*IQR;
  const double uppB = Q3 + 1.5*IQR;

  std::copy_if(data.begin(), data.end(), std::back_inserter(outliers),
    [&](std::chrono::duration<double> dur) { return dur.count() < lwrB || dur.count() > uppB;} );

  data.erase(std::remove_if(data.begin(), data.end(),
        [&](std::chrono::duration<double> dur) { return dur.count() < lwrB || dur.count() > uppB;} ),
      data.end());
}

void printVecInfo(const std::string &name, const std::vector<std::chrono::duration<double>> &data) {
  std::cout << name << "count: " << data.size() << "\n";
  for(auto &i : data)
    std::cout << "  " << i.count() << "\n";
}
