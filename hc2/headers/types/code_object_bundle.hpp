//===----------------------------------------------------------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "../functions/hsa_interfaces.hpp"

#include <hsa/hsa.h>

#include <algorithm>
#include <cstdint>
#include <istream>
#include <iterator>
#include <string>
#include <vector>

namespace hc2
{
    struct Bundled_code {
        union {
            struct {
                std::uint64_t offset;
                std::uint64_t bundle_sz;
                std::uint64_t triple_sz;
            };
            char cbuf[
                sizeof(offset) + sizeof(bundle_sz) + sizeof(triple_sz)];
        };
        std::string triple;
        std::vector<char> blob;
    };

    class Bundled_code_header {
        friend
        inline
        bool valid(const Bundled_code_header& x)
        {
            return std::equal(
                x.bundler_magic_string,
                x.bundler_magic_string + x.magic_string_sz,
                x.magic_string);
        }

        friend
        inline
        const std::vector<Bundled_code>& bundles(const Bundled_code_header& x)
        {
            return x.bundles;
        }

        template<typename RandomAccessIterator>
        friend
        inline
        bool read(
            RandomAccessIterator f,
            RandomAccessIterator l,
            Bundled_code_header& x)
        {
            std::copy_n(f, sizeof(x.cbuf), x.cbuf);

            if (valid(x)) {
                x.bundles.resize(x.bundle_cnt);

                auto it = f + sizeof(x.cbuf);
                for (auto&& y : x.bundles) {
                    std::copy_n(it, sizeof(y.cbuf), y.cbuf);
                    it += sizeof(y.cbuf);

                    y.triple.assign(it, it + y.triple_sz);

                    std::copy_n(
                        f + y.offset, y.bundle_sz, std::back_inserter(y.blob));

                    it += y.triple_sz;
                }

                return true;
            }

            return false;
        }

        friend
        inline
        bool read(const std::vector<char>& blob, Bundled_code_header& x)
        {
            return read(blob.cbegin(), blob.cend(), x);
        }

        friend
        inline
        bool read(std::istream& is, Bundled_code_header& x)
        {
            return read(std::vector<char>{
                std::istreambuf_iterator<char>{is},
                std::istreambuf_iterator<char>{}},
                x);
        }

        static constexpr const char magic_string[] = "__CLANG_OFFLOAD_BUNDLE__";
        static constexpr std::size_t magic_string_sz = sizeof(magic_string) - 1;

        union {
            struct {
                char bundler_magic_string[magic_string_sz];
                std::uint64_t bundle_cnt;
            };
            char cbuf[sizeof(bundler_magic_string) + sizeof(bundle_cnt)];
        };
        std::vector<Bundled_code> bundles;
    public:
        Bundled_code_header() = default;
        Bundled_code_header(const Bundled_code_header&) = default;
        Bundled_code_header(Bundled_code_header&&) = default;

        template<typename RandomAccessIterator>
        Bundled_code_header(RandomAccessIterator f, RandomAccessIterator l)
            : Bundled_code_header{}
        {
            read(f, l, *this);
        }

        explicit
        Bundled_code_header(const std::vector<char>& blob)
            : Bundled_code_header{blob.cbegin(), blob.cend()}
        {}
    };
    constexpr const char Bundled_code_header::magic_string[];

    inline
    std::string transmogrify_triple(const std::string& triple)
    {
        static constexpr const char old_prefix[]{"hcc-amdgcn--amdhsa-gfx"};
        static constexpr const char new_prefix[]{"hcc-amdgcn-amd-amdhsa--gfx"};

        if (triple.find(old_prefix) == 0) {
            return new_prefix + triple.substr(sizeof(old_prefix) - 1);
        }

        return (triple.find(new_prefix) == 0) ? triple : "";
    }

    inline
    std::string isa_name(std::string triple)
    {
        static constexpr const char offload_prefix[]{"hcc-"};
        const char* unsupported_gfx[] = {"801", "902"};

        triple = transmogrify_triple(triple);
        if (triple.empty()) return {};

        triple.erase(0, sizeof(offload_prefix) - 1);

        static hsa_isa_t tmp{};

        const char** gfx = unsupported_gfx;
        for (int i = 0; i < sizeof(unsupported_gfx)/sizeof(char*); ++i) {
            if (triple.find(*gfx) != std::string::npos)
                return {};
            ++gfx;
        }

        static const bool is_old_rocr{
            hsa_isa_from_name(triple.c_str(), &tmp) != HSA_STATUS_SUCCESS};

        if (is_old_rocr) {
             auto tmp{triple.substr(triple.rfind('x') + 1)};
            triple.replace(0, std::string::npos, "AMD:AMDGPU");

            for (auto&& x : tmp) {
                triple.push_back(':');
                triple.push_back(x);
           }
        }

         return triple;
    }

    inline
    hsa_isa_t triple_to_hsa_isa(std::string triple)
    {
        const auto isa{isa_name(std::move(triple))};

        if (isa.empty()) return hsa_isa_t({});

        hsa_isa_t r{};
        throwing_hsa_result_check(
            hsa_isa_from_name(isa.c_str(), &r), __FILE__, __func__, __LINE__);

        return r;
    }
}
