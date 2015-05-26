//#fail
#include <array_view>
int main()
{
    std::strided_array_view<char, 3> av;
    auto k = av[5566];
}
