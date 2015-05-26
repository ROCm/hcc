//#fail
#include <vector>
#include <array_view>
int main()
{
    std::array_view<int, 1> b;
    int a = b[0];
}
