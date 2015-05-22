//#fail
#include <array_view>
#include <vector>
int main()
{
    std::vector<int> v(10);
    std::array_view<int, 2> av(v, std::bounds<2>{100, 1});
}
