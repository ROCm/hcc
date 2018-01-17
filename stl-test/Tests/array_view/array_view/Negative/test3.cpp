//#error
#include <vector>
#include <array_view>
int main()
{
    std::vector<int> v(10);
    std::array_view<int, 2> av(v);
}
