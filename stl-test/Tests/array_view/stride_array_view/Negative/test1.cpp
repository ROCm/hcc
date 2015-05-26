//#error
#include <array_view>
int main()
{
    std::array_view<char, 3> av;
    std::strided_array_view<int, 3> sav(av);
}
