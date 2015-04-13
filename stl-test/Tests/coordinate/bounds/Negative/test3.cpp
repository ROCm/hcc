//#fail
#include <coordinate>
#include <cstddef>
int main(void)
{
    std::bounds<1> cord(std::numeric_limits<ptrdiff_t>::max() + 1);
}
