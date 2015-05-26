//#error
#include <vector>
#include <array_view>
int main()
{
    std::vector<float> v(10);
    std::array_view<int, 1> av(v);
}
