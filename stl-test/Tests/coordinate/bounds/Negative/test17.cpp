//#error
#include <coordinate>
int main(void)
{
    std::bounds<3> cord{55, 66, 77};
    auto cord /= -1;
}
