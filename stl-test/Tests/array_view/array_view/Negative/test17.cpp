//#fail
#include <vector>
#include <array_view>
int main()
{
    std::vector<int> v(10);
    std::array_view<int, 1> b(v, {2, 5});
    auto c = b[100];
}
