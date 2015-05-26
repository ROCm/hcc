//#error
#include <array_view>
int main()
{
    char arr[3][3];
    std::array_view<int, 2> av(arr);
}
