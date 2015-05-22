//#error
#include <array_view>
int main()
{
    std::array_view<int, 2> v;
    std::array_view<int, 3> av(v);
}
