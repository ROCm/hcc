//#error
#include <array_view>
int main()
{
    std::array_view<const char, 2> v;
    std::array_view<int, 2> av(v);
}
