//#error
#include <array_view>
int main()
{
    std::strided_array_view<char, 3> av;
    std::strided_array_view<int, 3> sav;
    sav = av;
}
