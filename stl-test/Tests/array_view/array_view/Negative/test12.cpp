//#fail
#include <vector>
#include <array_view>
int main()
{
    std::vector<int> vec(10);
    std::array_view<int, 1> b(vec);
    int a = b[{100}];
}
