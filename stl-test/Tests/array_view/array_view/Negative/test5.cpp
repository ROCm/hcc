//#error
#include <array_view>
int main()
{
    std::array_view<float, 2> v;
    std::array_view<int, 1> av(v);
}
