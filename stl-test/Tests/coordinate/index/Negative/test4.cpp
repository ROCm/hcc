//#fail
#include <coordinate>
int main(void)
{
    std::offset<2> cord{5566, 5566};
    auto value = cord[5566];
}
