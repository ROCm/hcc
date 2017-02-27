#if __cplusplus >= 201402L
    // RUN: %hc %s -o %t.out && %t.out
    #include <amp.h>

    #include <type_traits>

    bool generic_lambda_as_rvalue_is_valid_pfe_argument()
    {
        using namespace Concurrency;

        array_view<int> t{1};

        // Non-tiled domain.
        parallel_for_each(extent<1>{1}, [=](auto&& idx) restrict(amp) {
            static_assert(std::is_convertible<decltype(idx), index<1>>::value, "");

            t[idx[0]] = 1;
        });
        parallel_for_each(extent<2>{1, 1}, [=](auto&& idx) restrict(amp) {
            static_assert(std::is_convertible<decltype(idx), index<2>>::value, "");

            ++t[idx[0] + idx[1]];
        });
        parallel_for_each(extent<3>{1, 1, 1}, [=](auto&& idx) restrict(amp) {
            static_assert(std::is_convertible<decltype(idx), index<3>>::value, "");

            ++t[idx[0] + idx[1] + idx[2]];
        });

        // Tiled domain.
        parallel_for_each(extent<1>{1}.tile<1>(), [=](auto&& tidx) restrict(amp) {
            static_assert(std::is_convertible<decltype(tidx), tiled_index<1>>::value,
                          "");
            ++t[tidx.local[0]];
        });
        parallel_for_each(extent<2>{1, 1}.tile<1, 1>(), [=](auto&& tidx) restrict(amp) {
            static_assert(std::is_convertible<decltype(tidx), tiled_index<1, 1>>::value,
                          "");
            ++t[tidx.local[0] + tidx.local[1]];
        });
        parallel_for_each(extent<3>{1, 1, 1}.tile<1, 1, 1>(), [=](auto&& tidx) restrict(amp) {
            static_assert(std::is_convertible<decltype(tidx), tiled_index<1, 1, 1>>::value,
                          "");
            ++t[tidx.local[0] + tidx.local[1] + tidx.local[2]];
        });

        return t[0] == 6;
    }

    bool generic_lambda_as_lvalue_is_valid_pfe_argument()
    {
        using namespace Concurrency;

        array_view<int> t{1}; t[0] = 0;

        const auto lambda = [=](auto&& idx) restrict(amp) {
            #if __cplusplus > 201402L
                if constexpr(std::is_same_v<std::decay_t<decltype(idx)>,
                                            index<1>>) {
                    ++t[idx[0]];
                }
                else if constexpr(std::is_same_v<std::decay_t<decltype(idx)>,
                                                 index<2>>) {
                    ++t[idx[0] + idx[1]];
                }
                else if constexpr(std::is_same_v<std::decay_t<decltype(idx)>,
                                                 index<3>>) {
                    ++t[idx[0] + idx[1] + idx[2]];
                }
                else if constexpr(std::is_same_v<std::decay_t<decltype(idx)>,
                                                 tiled_index<1>>) {
                    ++t[idx.local[0]];
                }
                else if constexpr(std::is_same_v<std::decay_t<decltype(idx)>,
                                                 tiled_index<1, 1>>) {
                    ++t[idx.local[0] + idx.local[1]];
                }
                else ++t[idx.local[0] + idx.local[1] + idx.local[2]];
            #else
                ++t[0];
            #endif
        };

        // Non-tiled domain.
        parallel_for_each(extent<1>{1}, lambda);
        parallel_for_each(extent<3>{1, 1, 1}, lambda);
        parallel_for_each(extent<2>{1, 1}, lambda);

        // Tiled domain.
        parallel_for_each(extent<1>{1}.tile<1>(), lambda);
        parallel_for_each(extent<2>{1, 1}.tile<1, 1>(), lambda);
        parallel_for_each(extent<3>{1, 1, 1}.tile<1, 1, 1>(), lambda);

        return t[0] == 6;
    }

    int main()
    {
        // TODO: this is temporary, and not a proper unit test.
        if (!generic_lambda_as_rvalue_is_valid_pfe_argument()) return EXIT_FAILURE;
        if (!generic_lambda_as_lvalue_is_valid_pfe_argument()) return EXIT_FAILURE;
        return EXIT_SUCCESS;
    }
#endif