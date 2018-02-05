//#fail
#include <array_view>
int main()
{
    std::strided_array_view<char, 3> av;
    char a = av[{1, 1, 1}];
}
