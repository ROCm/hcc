//#fail
#include <vector>
#include <array_view>
int main()
{
    std::vector<int> vec(10);
    std::array_view<int, 2> b(vec, {2, 5});
    int a = b[{100, 100}];
}
