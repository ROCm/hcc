// RUN: %cxxamp %s %link
// RUN: ./a.out
#include <amp.h>
#include "../../../accelerator.common.h"

using namespace Concurrency;

int main()
{
    int result = 1;

    accelerator acc;
    accelerator acc_expected(accelerator::default_accelerator);

    const wchar_t literal_string[256] = L"default";

    accelerator acc_literal_string(static_cast<const wchar_t*>(literal_string));

    result &= is_accelerator_equal(acc, acc_expected);
    result &= is_accelerator_view_operable(acc.get_default_view());
    result &= is_accelerator_equal(acc, acc_literal_string);
    result &= is_accelerator_view_operable(acc_literal_string.get_default_view());

    return !result;
}
