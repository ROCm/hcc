//#error
#include <array_view>
int main()
{
    const std::array_view<unsigned int, 2> v;
    std::array_view<int, 1> av(v);
}
