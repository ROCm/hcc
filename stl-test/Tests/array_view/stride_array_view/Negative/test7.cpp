//#fail
#include <array_view>
#include <vector>
int main()
{
    std::vector<char> v(900);
    std::array_view<char, 4> aav(v, {5, 5, 6, 6});
    std::strided_array_view<char, 4> av(aav);
    auto c = av.section({2, 2, 6, 6}, {1, 1, 1, 1});
}
