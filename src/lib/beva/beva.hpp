#pragma once

#include <vector>
#include <string>
#include <format>
#include <array>
#include <unordered_map>
#include <memory>
#include <any>
#include <optional>
#include <limits>
#include <type_traits>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <cstdint>

#include "vulkan/vulkan.h"
#include "vulkan/vk_enum_string_helper.h"

#define _BV_DELETE_DEFAULT_CTOR(ClassName) ClassName() = delete

#define _BV_ALLOW_MOVE_ONLY(ClassName) \
ClassName(const ClassName& other) = delete; \
ClassName& operator=(const ClassName& other) = delete; \
ClassName(ClassName&& other) = default; \
ClassName& operator=(ClassName&& other) = default

#define _BV_DELETE_DEFAULT_CTOR_AND_ALLOW_MOVE_ONLY(ClassName) \
_BV_DELETE_DEFAULT_CTOR(ClassName); _BV_ALLOW_MOVE_ONLY(ClassName)

#define _BV_DEFINE_SMART_PTR_TYPE_ALIASES(ClassName) \
    using ClassName##Ptr = std::shared_ptr<ClassName>; \
    using ClassName##WPtr = std::weak_ptr<ClassName>

#pragma region dynamic_bitset

/*

this region contains a copy of dynamic_bitset by https://github.com/pinam45
source: https://github.com/pinam45/dynamic_bitset/blob/ac60c9e6c534db7457ca6af02fdedbe74ad60968/include/sul/dynamic_bitset.hpp

the following comment contains the license for the dynamic_bitset library.

*/

/*

MIT License

Copyright (c) 2019 Maxime Pinard

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#define DYNAMIC_BITSET_NO_LIBPOPCNT

//
// Copyright (c) 2019 Maxime Pinard
//
// Distributed under the MIT license
// See accompanying file LICENSE or copy at
// https://opensource.org/licenses/MIT
//
#ifndef SUL_DYNAMIC_BITSET_HPP
#define SUL_DYNAMIC_BITSET_HPP

/**
 * @brief      @ref sul::dynamic_bitset version major.
 *
 * @since      1.1.0
 */
#define SUL_DYNAMIC_BITSET_VERSION_MAJOR 1

 /**
  * @brief      @ref sul::dynamic_bitset version minor.
  *
  * @since      1.1.0
  */
#define SUL_DYNAMIC_BITSET_VERSION_MINOR 3

  /**
   * @brief      @ref sul::dynamic_bitset version patch.
   *
   * @since      1.1.0
   */
#define SUL_DYNAMIC_BITSET_VERSION_PATCH 0

   /** @file
    * @brief      @ref sul::dynamic_bitset declaration and implementation.
    *
    * @details    Standalone file, does not depend on other implementation files or dependencies other
    *             than the standard library, can be taken and used directly.
    *
    *             Can optionally include and use libpopcnt if @a DYNAMIC_BITSET_NO_LIBPOPCNT is not
    *             defined and @a __has_include(\<libpopcnt.h\>) is @a true.
    *
    * @remark     Include multiple standard library headers and optionally @a libpopcnt.h.
    *
    * @since      1.0.0
    */

#include <memory>
#include <vector>
#include <algorithm>
#include <string>
#include <string_view>
#include <functional>
#include <type_traits>
#include <limits>
#include <stdexcept>
#include <cmath>
#include <cassert>

    // define DYNAMIC_BITSET_CAN_USE_LIBPOPCNT
#if !defined(DYNAMIC_BITSET_NO_LIBPOPCNT)
// https://github.com/kimwalisch/libpopcnt
#	if __has_include(<libpopcnt.h>)
#		include <libpopcnt.h>
#		define DYNAMIC_BITSET_CAN_USE_LIBPOPCNT true
#	endif
#endif
#if !defined(DYNAMIC_BITSET_CAN_USE_LIBPOPCNT)
#	define DYNAMIC_BITSET_CAN_USE_LIBPOPCNT false
#endif

// define DYNAMIC_BITSET_CAN_USE_STD_BITOPS
#if !defined(DYNAMIC_BITSET_NO_STD_BITOPS)
// https://en.cppreference.com/w/cpp/header/bit
#	if __has_include(<bit>)
#		include <bit>
#		ifdef __cpp_lib_bitops
#			define DYNAMIC_BITSET_CAN_USE_STD_BITOPS true
#		endif
#	endif
#endif
#if !defined(DYNAMIC_BITSET_CAN_USE_STD_BITOPS)
#	define DYNAMIC_BITSET_CAN_USE_STD_BITOPS false
#endif

// define DYNAMIC_BITSET_CAN_USE_CLANG_BUILTIN_POPCOUNT
// define DYNAMIC_BITSET_CAN_USE_CLANG_BUILTIN_CTZ
// define DYNAMIC_BITSET_CAN_USE_GCC_BUILTIN
// define DYNAMIC_BITSET_CAN_USE_MSVC_BUILTIN_BITSCANFORWARD
// define DYNAMIC_BITSET_CAN_USE_MSVC_BUILTIN_BITSCANFORWARD64
#if !DYNAMIC_BITSET_CAN_USE_STD_BITOPS && !defined(DYNAMIC_BITSET_NO_COMPILER_BUILTIN)
#	if defined(__clang__)
// https://clang.llvm.org/docs/LanguageExtensions.html#feature-checking-macros
// https://clang.llvm.org/docs/LanguageExtensions.html#intrinsics-support-within-constant-expressions
#		ifdef __has_builtin
#			if __has_builtin(__builtin_popcount) && __has_builtin(__builtin_popcountl) \
			  && __has_builtin(__builtin_popcountll)
#				define DYNAMIC_BITSET_CAN_USE_CLANG_BUILTIN_POPCOUNT true
#			endif
#			if __has_builtin(__builtin_ctz) && __has_builtin(__builtin_ctzl) \
			  && __has_builtin(__builtin_ctzll)
#				define DYNAMIC_BITSET_CAN_USE_CLANG_BUILTIN_CTZ true
#			endif
#		endif
#	elif defined(__GNUC__) // also defined by clang
// https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html
#		define DYNAMIC_BITSET_CAN_USE_GCC_BUILTIN true
#	elif defined(_MSC_VER)
// https://docs.microsoft.com/en-us/cpp/intrinsics/bitscanforward-bitscanforward64
// __popcnt16, __popcnt, __popcnt64 not used because it require to check the hardware support at runtime
// (https://docs.microsoft.com/fr-fr/cpp/intrinsics/popcnt16-popcnt-popcnt64?view=msvc-160#remarks)
#		if defined(_M_IX86) || defined(_M_ARM) || defined(_M_X64) || defined(_M_ARM64)
#			include <intrin.h>
#			pragma intrinsic(_BitScanForward)
#			define DYNAMIC_BITSET_CAN_USE_MSVC_BUILTIN_BITSCANFORWARD true
#		endif
#		if(defined(_M_X64) || defined(_M_ARM64)) \
		  && !defined(DYNAMIC_BITSET_NO_MSVC_BUILTIN_BITSCANFORWARD64) // for testing purposes
#			pragma intrinsic(_BitScanForward64)
#			define DYNAMIC_BITSET_CAN_USE_MSVC_BUILTIN_BITSCANFORWARD64 true
#		endif
#	endif
#endif
#if !defined(DYNAMIC_BITSET_CAN_USE_CLANG_BUILTIN_POPCOUNT)
#	define DYNAMIC_BITSET_CAN_USE_CLANG_BUILTIN_POPCOUNT false
#endif
#if !defined(DYNAMIC_BITSET_CAN_USE_CLANG_BUILTIN_CTZ)
#	define DYNAMIC_BITSET_CAN_USE_CLANG_BUILTIN_CTZ false
#endif
#if !defined(DYNAMIC_BITSET_CAN_USE_GCC_BUILTIN)
#	define DYNAMIC_BITSET_CAN_USE_GCC_BUILTIN false
#endif
#if !defined(DYNAMIC_BITSET_CAN_USE_MSVC_BUILTIN_BITSCANFORWARD)
#	define DYNAMIC_BITSET_CAN_USE_MSVC_BUILTIN_BITSCANFORWARD false
#endif
#if !defined(DYNAMIC_BITSET_CAN_USE_MSVC_BUILTIN_BITSCANFORWARD64)
#	define DYNAMIC_BITSET_CAN_USE_MSVC_BUILTIN_BITSCANFORWARD64 false
#endif
#if !defined(DYNAMIC_BITSET_CAN_USE_MSVC_BUILTIN)
#	define DYNAMIC_BITSET_CAN_USE_MSVC_BUILTIN false
#endif

#ifndef DYNAMIC_BITSET_NO_NAMESPACE
/**
 * @brief      Simple Useful Libraries.
 *
 * @since      1.0.0
 */
namespace sul
{
#endif

    /**
     * @brief      Dynamic bitset.
     *
     * @details    Data structure used to store a vector of bits and apply binary operations to it. The
     *             bits are stored in an optimized way in an underling block type. It is highly inspired
     *             by std\::bitset but with a run-time changeable size.
     *
     *             Preconditions are checked with @a assert but no exception will be thrown if one is
     *             violated (as with std\::bitset).
     *
     * @remark     It is not a Container as it does not provide iterators because of the reference proxy
     *             class used to access the bits.
     *
     * @tparam     Block      Block type to use for storing the bits, must be an unsigned integral type
     * @tparam     Allocator  Allocator type to use for memory management, must meet the standard
     *                        requirements of @a Allocator
     *
     * @since      1.0.0
     */
    template<typename Block = unsigned long long, typename Allocator = std::allocator<Block>>
    class dynamic_bitset
    {
        static_assert(std::is_unsigned<Block>::value, "Block is not an unsigned integral type");

    public:
        /**
         * @brief      Type used to represent the size of a @ref sul::dynamic_bitset.
         *
         * @since      1.0.0
         */
        typedef size_t size_type;

        /**
         * @brief      Same type as @p Block.
         *
         * @since      1.0.0
         */
        typedef Block block_type;

        /**
         * @brief      Same type as @p Allocator.
         *
         * @since      1.0.0
         */
        typedef Allocator allocator_type;

        /**
         * @brief      Number of bits that can be stored in a block.
         *
         * @since      1.0.0
         */
        static constexpr size_type bits_per_block = std::numeric_limits<block_type>::digits;

        /**
         * @brief      Maximum value of @ref size_type, returned for invalid positions.
         *
         * @since      1.0.0
         */
        static constexpr size_type npos = std::numeric_limits<size_type>::max();

        /**
         * @brief      Reference to a @ref sul::dynamic_bitset bit.
         *
         * @details    As the bits in the @ref sul::dynamic_bitset class are stored in an optimized way
         *             in blocks, it is not possible for the subscript operators to return a reference
         *             to a boolean. Hence this class is used as a proxy to enable subscript operator of
         *             the @ref sul::dynamic_bitset class to be used as if it was an array of bools.
         *
         * @since      1.0.0
         */
        class reference
        {
        public:
            /**
             * @brief      Constructs a @ref reference to a bit from a @ref sul::dynamic_bitset and a
             *             bit position.
             *
             * @param      bitset   @ref sul::dynamic_bitset containing the bit
             * @param[in]  bit_pos  Position of the bit to reference in the @ref sul::dynamic_bitset
             *
             * @complexity Constant.
             *
             * @since      1.0.0
             */
            constexpr reference(dynamic_bitset<Block, Allocator>& bitset, size_type bit_pos);

            /**
             * @brief      Copy constructor.
             *
             * @since      1.0.0
             */
            constexpr reference(const reference&) noexcept = default;

            /**
             * @brief      Move constructor.
             *
             * @since      1.0.0
             */
            constexpr reference(reference&&) noexcept = default;

            /**
             * @brief      Destructor.
             *
             * @since      1.0.0
             */
            ~reference() noexcept = default;

            /**
             * @brief      Assign a value to the referenced bit.
             *
             * @param[in]  v     Value to assign to the referenced bit
             *
             * @return     The @ref reference
             *
             * @complexity Constant.
             *
             * @since      1.0.0
             */
            constexpr reference& operator=(bool v);

            /**
             * @brief      Assign a value to the referenced bit from another @ref reference.
             *
             * @param[in]  rhs   @ref reference to the bit to assign value from
             *
             * @return     The @ref reference
             *
             * @complexity Constant.
             *
             * @since      1.0.0
             */
            constexpr reference& operator=(const reference& rhs);

            /**
             * @brief      Assign a value to the referenced bit from another @ref reference.
             *
             * @param[in]  rhs   @ref reference to the bit to assign value from
             *
             * @return     The @ref reference
             *
             * @complexity Constant.
             *
             * @since      1.0.0
             */
            constexpr reference& operator=(reference&& rhs) noexcept;

            /**
             * @brief      Apply binary operator AND to the referenced bit and a value, and assign the
             *             result to the referenced bit.
             *
             * @param[in]  v     Value to apply binary operator AND with
             *
             * @return     The @ref reference
             *
             * @complexity Constant.
             *
             * @since      1.0.0
             */
            constexpr reference& operator&=(bool v);

            /**
             * @brief      Apply binary operator OR to the referenced bit and a value, and assign the
             *             result to the referenced bit.
             *
             * @param[in]  v     Value to apply binary operator OR with
             *
             * @return     The @ref reference
             *
             * @complexity Constant.
             *
             * @since      1.0.0
             */
            constexpr reference& operator|=(bool v);

            /**
             * @brief      Apply binary operator XOR to the referenced bit and a value, and assign the
             *             result to the referenced bit.
             *
             * @param[in]  v     Value to apply binary operator XOR with
             *
             * @return     The @ref reference
             *
             * @complexity Constant.
             *
             * @since      1.0.0
             */
            constexpr reference& operator^=(bool v);

            /**
             * @brief      Apply binary difference to the referenced bit and a value, and assign the
             *             result to the referenced bit.
             *
             * @details    Equivalent to:
             *             @code
             *             this &= !v;
             *             @endcode
             *
             * @param[in]  v     Value to apply binary difference with
             *
             * @return     The @ref reference
             *
             * @complexity Constant.
             *
             * @since      1.0.0
             */
            constexpr reference& operator-=(bool v);

            /**
             * @brief      Return the result of applying unary NOT operator.
             *
             * @return     The opposite of the referenced bit
             *
             * @complexity Constant.
             *
             * @since      1.0.0
             */
            [[nodiscard]] constexpr bool operator~() const;

            /**
             * @brief      bool conversion operator.
             *
             * @complexity Constant.
             *
             * @since      1.0.0
             */
            [[nodiscard]] constexpr operator bool() const;

            /**
             * @brief      Deleted to avoid taking the address of a temporary proxy object.
             *
             * @since      1.0.0
             */
            constexpr void operator&() = delete;

            /**
             * @brief      Set the referenced bit to @a true.
             *
             * @return     The @ref reference
             *
             * @complexity Constant.
             *
             * @since      1.0.0
             */
            constexpr reference& set();

            /**
             * @brief      Reset the referenced bit to @a false.
             *
             * @return     The @ref reference
             *
             * @complexity Constant.
             *
             * @since      1.0.0
             */
            constexpr reference& reset();

            /**
             * @brief      Flip the referenced bit.
             *
             * @return     The @ref reference
             *
             * @complexity Constant.
             *
             * @since      1.0.0
             */
            constexpr reference& flip();

            /**
             * @brief      Assign the value @p v to the referenced bit.
             *
             * @param[in]  v     Value to assign to the bit
             *
             * @return     The @ref reference
             *
             * @complexity Constant.
             *
             * @since      1.0.0
             */
            constexpr reference& assign(bool v);

        private:
            block_type& m_block;
            block_type m_mask;
        };

        /**
         * @brief      Const reference to a @ref sul::dynamic_bitset bit, type bool.
         *
         * @since      1.0.0
         */
        typedef bool const_reference;

        /**
         * @brief      Copy constructor.
         *
         * @since      1.0.0
         */
        constexpr dynamic_bitset(const dynamic_bitset<Block, Allocator>& other) = default;

        /**
         * @brief      Move constructor.
         *
         * @since      1.0.0
         */
        constexpr dynamic_bitset(dynamic_bitset<Block, Allocator>&& other) noexcept = default;

        /**
         * @brief      Copy assignment operator.
         *
         * @since      1.0.0
         */
        constexpr dynamic_bitset<Block, Allocator>& operator=(
            const dynamic_bitset<Block, Allocator>& other) = default;

        /**
         * @brief      Move assignment operator.
         *
         * @since      1.0.0
         */
        constexpr dynamic_bitset<Block, Allocator>& operator=(
            dynamic_bitset<Block, Allocator>&& other) noexcept = default;

        /**
         * @brief      Constructs an empty @ref sul::dynamic_bitset.
         *
         * @details    A copy of @p allocator will be used for memory management.
         *
         * @param[in]  allocator  Allocator to use for memory management
         *
         * @complexity Constant.
         *
         * @since      1.0.0
         */
        constexpr explicit dynamic_bitset(const allocator_type& allocator = allocator_type());

        /**
         * @brief      Constructs a @ref sul::dynamic_bitset of @p nbits bits from an initial value.
         *
         * @details    The first bits are initialized with the bits from @p init_val, if @p nbits \>
         *             std\::numeric_limits\<unsigned long long\>\::digits , all other bits are
         *             initialized to @a false. A copy of @p allocator will be used for memory
         *             management.
         *
         * @param[in]  nbits      Number of bits of the @ref sul::dynamic_bitset
         * @param[in]  init_val   Value to initialize the @ref sul::dynamic_bitset with
         * @param[in]  allocator  Allocator to use for memory management
         *
         * @complexity Linear in @p nbits / @ref bits_per_block.
         *
         * @since      1.0.0
         */
        constexpr explicit dynamic_bitset(size_type nbits,
            unsigned long long init_val = 0,
            const allocator_type& allocator = allocator_type());

        /**
         * @brief      Constructs a @ref sul::dynamic_bitset using @p init_vals to initialize the first
         *             blocks.
         *
         * @details    The size of the newly created @ref sul::dynamic_bitset is @p init_vals.size() *
         *             @ref bits_per_block. A copy of @p allocator will be used for memory management.
         *
         * @param[in]  init_vals  Value of the @ref sul::dynamic_bitset first blocks
         * @param[in]  allocator  Allocator to use for memory management
         *
         * @complexity Linear in @p init_vals.size().
         *
         * @since      1.0.0
         */
        constexpr dynamic_bitset(std::initializer_list<block_type> init_vals,
            const allocator_type& allocator = allocator_type());

        /**
         * @brief      Constructs a @ref sul::dynamic_bitset from a string or a part of a string.
         *
         * @details    Construct the @ref sul::dynamic_bitset using the characters from @p str in the
         *             range \[@p pos, std\::min(@p pos + @p n, @p str.size())\[.
         *
         * @param[in]  str        String containing the part to use
         * @param[in]  pos        Starting position of the string part to use in @p str
         * @param[in]  n          Number of characters of @p str to use from the starting position
         * @param[in]  zero       Character used to represent @a false bits in @p str
         * @param[in]  one        Character used to represent @a true bits in @p str
         * @param[in]  allocator  Allocator to use for memory management
         *
         * @tparam     _CharT     Character type of the string
         * @tparam     _Traits    Traits class specifying the operations on the character type of the
         *                        string
         *
         * @pre        @code
         *             pos < str.size()
         *             @endcode
         *
         * @complexity Linear in std\::min(@p n, @p str.size() - @p pos).
         *
         * @since      1.0.0
         */
        template<typename _CharT, typename _Traits>
        constexpr explicit dynamic_bitset(
            std::basic_string_view<_CharT, _Traits> str,
            typename std::basic_string_view<_CharT, _Traits>::size_type pos = 0,
            typename std::basic_string_view<_CharT, _Traits>::size_type n =
            std::basic_string_view<_CharT, _Traits>::npos,
            _CharT zero = _CharT('0'),
            _CharT one = _CharT('1'),
            const allocator_type& allocator = allocator_type());

        /**
         * @brief      Constructs a @ref sul::dynamic_bitset from a string or a part of a string.
         *
         * @details    Construct the @ref sul::dynamic_bitset using the characters from @p str in the
         *             range \[@p pos, std\::min(@p pos + @p n, @p str.size())\[.
         *
         * @param[in]  str        String containing the part to use
         * @param[in]  pos        Starting position of the string part to use in @p str
         * @param[in]  n          Number of characters of @p str to use from the starting position
         * @param[in]  zero       Character used to represent @a false bits in @p str
         * @param[in]  one        Character used to represent @a true bits in @p str
         * @param[in]  allocator  Allocator to use for memory management
         *
         * @tparam     _CharT     Character type of the string
         * @tparam     _Traits    Traits class specifying the operations on the character type of the
         *                        string
         * @tparam     _Alloc     Allocator type used to allocate internal storage of the string
         *
         * @pre        @code
         *             pos < str.size()
         *             @endcode
         *
         * @complexity Linear in std\::min(@p n, @p str.size() - @p pos).
         *
         * @since      1.0.0
         */
        template<typename _CharT, typename _Traits, typename _Alloc>
        constexpr explicit dynamic_bitset(
            const std::basic_string<_CharT, _Traits, _Alloc>& str,
            typename std::basic_string<_CharT, _Traits, _Alloc>::size_type pos = 0,
            typename std::basic_string<_CharT, _Traits, _Alloc>::size_type n =
            std::basic_string<_CharT, _Traits, _Alloc>::npos,
            _CharT zero = _CharT('0'),
            _CharT one = _CharT('1'),
            const allocator_type& allocator = allocator_type());

        /**
         * @brief      Constructs a @ref sul::dynamic_bitset from a string or a part of a string.
         *
         * @details    Construct the @ref sul::dynamic_bitset using the characters from @p str in the
         *             range \[@p pos, std\::min(@p pos + @p n, @p _Traits\::length(@p str))\[.
         *
         * @param[in]  str        String containing the part to use
         * @param[in]  pos        Starting position of the string part to use
         * @param[in]  n          Number of characters to use from the starting position
         * @param[in]  zero       Character used to represent @a false bits in the string
         * @param[in]  one        Character used to represent 1 @a true bitsn the string
         * @param[in]  allocator  Allocator to use for memory management
         *
         * @tparam     _CharT     Character type of the string
         * @tparam     _Traits    Traits class specifying the operations on the character type of the
         *                        string
         *
         * @pre        @code
         *             pos < _Traits::length(str)
         *             @endcode
         *
         * @complexity Linear in std\::min(@p n, @p _Traits\::length(@p str) - @p pos).
         *
         * @since      1.0.0
         */
        template<typename _CharT, typename _Traits = std::char_traits<_CharT>>
        constexpr explicit dynamic_bitset(
            const _CharT* str,
            typename std::basic_string<_CharT>::size_type pos = 0,
            typename std::basic_string<_CharT>::size_type n = std::basic_string<_CharT>::npos,
            _CharT zero = _CharT('0'),
            _CharT one = _CharT('1'),
            const allocator_type& allocator = allocator_type());

        /**
         * @brief      Destructor.
         *
         * @since      1.0.0
         */
        ~dynamic_bitset() noexcept = default;

        /**
         * @brief      Resize the @ref sul::dynamic_bitset to contain @p nbits bits.
         *
         * @details    Bits keep the value they had before the resize and, if @p nbits is greater than
         *             the current size, new bit are initialized to @p value.
         *
         * @param[in]  nbits  New size of the @ref sul::dynamic_bitset
         * @param[in]  value  Value of the new bits
         *
         * @complexity Linear in the difference between the current size and @p nbits.
         *             Additional complexity possible due to reallocation if capacity is less than @p
         *             nbits.
         *
         * @since      1.0.0
         */
        constexpr void resize(size_type nbits, bool value = false);

        /**
         * @brief      Clears the @ref sul::dynamic_bitset, resize it to 0.
         *
         * @details    Equivalent to:
         *             @code
         *             this.resize(0);
         *             @endcode
         *
         * @complexity Linear in the size of the @ref sul::dynamic_bitset.
         *
         * @since      1.0.0
         */
        constexpr void clear();

        /**
         * @brief      Add a new bit with the value @p value at the end of the @ref sul::dynamic_bitset.
         *
         * @details    Increase the size of the bitset by one, the added bit becomes the
         *             most-significant bit.
         *
         * @param[in]  value  Value of the bit to add
         *
         * @complexity Amortized constant.
         *
         * @since      1.0.0
         */
        constexpr void push_back(bool value);

        /**
         * @brief      Remove the last bit of the @ref sul::dynamic_bitset.
         *
         * @details    Decrease the size of the @ref sul::dynamic_bitset by one, does nothing if the
         *             @ref dynamic_bitset is empty.
         *
         * @complexity Constant.
         *
         * @since      1.0.0
         */
        constexpr void pop_back();

        /**
         * @brief      Append a block of bits @p block at the end of the @ref sul::dynamic_bitset.
         *
         * @details    Increase the size of the @ref sul::dynamic_bitset by @ref bits_per_block.
         *
         * @param[in]  block  Block of bits to add
         *
         * @complexity Amortized constant.
         *
         * @since      1.0.0
         */
        constexpr void append(block_type block);

        /**
         * @brief      Append blocks of bits from @p blocks at the end of the @ref sul::dynamic_bitset.
         *
         * @param[in]  blocks  Blocks of bits to add
         *
         * @complexity Linear in the size of @p blocks. Additional complexity possible due
         *             to reallocation if capacity is less than @ref size() + @p blocks.size() * @ref
         *             bits_per_block.
         *
         * @since      1.0.0
         */
        constexpr void append(std::initializer_list<block_type> blocks);

        /**
         * @brief      Append blocks of bits from the range \[@p first, @p last\[ at the end of the @ref
         *             dynamic_bitset.
         *
         * @param[in]  first               First iterator of the range
         * @param[in]  last                Last iterator of the range (after the last element to add)
         *
         * @tparam     BlockInputIterator  Type of the range iterators
         *
         * @complexity Linear in the size of the range. Additional complexity possible due
         *             to reallocation if capacity is less than @ref size() + std\::distance(@p first,
         *             @p last) * @ref bits_per_block.
         *
         * @since      1.0.0
         */
        template<typename BlockInputIterator>
        constexpr void append(BlockInputIterator first, BlockInputIterator last);

        /**
         * @brief      Sets the bits to the result of binary AND on corresponding pairs of bits of *this
         *             and @p rhs.
         *
         * @param[in]  rhs   Right hand side @ref sul::dynamic_bitset of the operator
         *
         * @return     A reference to the @ref sul::dynamic_bitset *this
         *
         * @pre       @code
         *             size() == rhs.size()
         *             @endcode
         *
         * @complexity Linear in the size of the @ref sul::dynamic_bitset.
         *
         * @since      1.0.0
         */
        constexpr dynamic_bitset<Block, Allocator>& operator&=(
            const dynamic_bitset<Block, Allocator>& rhs);

        /**
         * @brief      Sets the bits to the result of binary OR on corresponding pairs of bits of *this
         *             and @p rhs.
         *
         * @param[in]  rhs   Right hand side @ref sul::dynamic_bitset of the operator
         *
         * @return     A reference to the @ref sul::dynamic_bitset *this
         *
         * @pre        @code
         *             size() == rhs.size()
         *             @endcode
         *
         * @complexity Linear in the size of the @ref sul::dynamic_bitset.
         *
         * @since      1.0.0
         */
        constexpr dynamic_bitset<Block, Allocator>& operator|=(
            const dynamic_bitset<Block, Allocator>& rhs);

        /**
         * @brief      Sets the bits to the result of binary XOR on corresponding pairs of bits of *this
         *             and @p rhs.
         *
         * @param[in]  rhs   Right hand side @ref sul::dynamic_bitset of the operator
         *
         * @return     A reference to the @ref sul::dynamic_bitset *this
         *
         * @pre        @code
         *             size() == rhs.size()
         *             @endcode
         *
         * @complexity Linear in the size of the @ref sul::dynamic_bitset.
         *
         * @since      1.0.0
         */
        constexpr dynamic_bitset<Block, Allocator>& operator^=(
            const dynamic_bitset<Block, Allocator>& rhs);

        /**
         * @brief      Sets the bits to the result of the binary difference between the bits of *this
         *             and @p rhs.
         *
         * @details    Less efficient but equivalent way to get this result:
         *             @code
         *             this &= ~rhs;
         *             @endcode
         *
         * @param[in]  rhs   Right hand side @ref sul::dynamic_bitset of the operator
         *
         * @return     A reference to the @ref sul::dynamic_bitset *this
         *
         * @pre        @code
         *             size() == rhs.size()
         *             @endcode
         *
         * @complexity Linear in the size of the @ref sul::dynamic_bitset.
         *
         * @since      1.0.0
         */
        constexpr dynamic_bitset<Block, Allocator>& operator-=(
            const dynamic_bitset<Block, Allocator>& rhs);

        /**
         * @brief      Performs binary shift left of @p shift bits.
         *
         * @details    Zeroes are shifted in, does nothing if @p shift == 0.
         *
         * @param[in]  shift  Number of positions to shift the bits
         *
         * @return     A reference to the @ref sul::dynamic_bitset *this
         *
         * @complexity Linear in the size of the @ref sul::dynamic_bitset.
         *
         * @since      1.0.0
         */
        constexpr dynamic_bitset<Block, Allocator>& operator<<=(size_type shift);

        /**
         * @brief      Performs binary shift right of @p shift bits.
         *
         * @details    Zeroes are shifted in, does nothing if @p shift == 0.
         *
         * @param[in]  shift  Number of positions to shift the bits
         *
         * @return     A reference to the @ref sul::dynamic_bitset *this
         *
         * @complexity Linear in the size of the @ref sul::dynamic_bitset.
         *
         * @since      1.0.0
         */
        constexpr dynamic_bitset<Block, Allocator>& operator>>=(size_type shift);

        /**
         * @brief      Performs binary shift right of @p shift bits.
         *
         * @details    Zeroes are shifted in. Does nothing if @p shift == 0.\n
         *             Equivalent to:
         *             @code
         *             dynamic_bitset<Block, Allocator> bitset(*this);
         *             bitset <<= shift;
         *             @endcode
         *
         * @param[in]  shift  Number of positions to shift the bits
         *
         * @return     A new @ref sul::dynamic_bitset containing the shifted bits
         *
         * @complexity Linear in the size of the @ref sul::dynamic_bitset.
         *
         * @since      1.0.0
         */
        [[nodiscard]] constexpr dynamic_bitset<Block, Allocator> operator<<(size_type shift) const;

        /**
         * @brief      Performs binary shift left of @p shift bits.
         *
         * @details    Zeroes are shifted in. Does nothing if @p shift == 0.\n
         *             Equivalent to:
         *             @code
         *             dynamic_bitset<Block, Allocator> bitset(*this);
         *             bitset >>= shift;
         *             @endcode
         *
         * @param[in]  shift  Number of positions to shift the bits
         *
         * @return     A new @ref sul::dynamic_bitset containing the shifted bits
         *
         * @complexity Linear in the size of the @ref sul::dynamic_bitset.
         *
         * @since      1.0.0
         */
        [[nodiscard]] constexpr dynamic_bitset<Block, Allocator> operator>>(size_type shift) const;

        /**
         * @brief      Performs a unary NOT on all bits.
         *
         * @details    Equivalent to:
         *             @code
         *             dynamic_bitset<Block, Allocator> bitset(*this);
         *             bitset.flip();
         *             @endcode
         *
         * @return     A copy of *this with all bits flipped
         *
         * @complexity Linear in the size of the @ref sul::dynamic_bitset.
         *
         * @since      1.0.0
         */
        [[nodiscard]] constexpr dynamic_bitset<Block, Allocator> operator~() const;

        /**
         * @brief      Set the bits of the range \[@p pos, @p pos + @p len\[ to value @p value.
         *
         * @details    Does nothing if @p len == 0.
         *
         * @param[in]  pos    Position of the first bit of the range
         * @param[in]  len    Length of the range
         * @param[in]  value  Value to set the bits to
         *
         * @return     A reference to the @ref sul::dynamic_bitset *this
         *
         * @pre        @code
         *             (pos < size()) && ((len == 0) || (pos + len - 1 < size()))
         *             @endcode
         *
         * @complexity Linear in @p len.
         *
         * @since      1.0.0
         */
        constexpr dynamic_bitset<Block, Allocator>& set(size_type pos, size_type len, bool value);

        /**
         * @brief      Set the bit at the position @p pos to @a true or value @p value.
         *
         * @param[in]  pos    Position of the bit to set
         * @param[in]  value  Value to set the bit to
         *
         * @return     A reference to the @ref sul::dynamic_bitset *this
         *
         * @pre        @code
         *             pos < size()
         *             @endcode
         *
         * @complexity Constant.
         *
         * @since      1.0.0
         */
        constexpr dynamic_bitset<Block, Allocator>& set(size_type pos, bool value = true);

        /**
         * @brief      Set all the bits of the @ref sul::dynamic_bitset to @a true.
         *
         * @return     A reference to the @ref sul::dynamic_bitset *this
         *
         * @complexity Linear in the size of the @ref sul::dynamic_bitset.
         *
         * @since      1.0.0
         */
        constexpr dynamic_bitset<Block, Allocator>& set();

        /**
         * @brief      Reset the bits of the range \[@p pos, @p pos + @p len\[ to @a false.
         *
         * @param[in]  pos   Position of the first bit of the range
         * @param[in]  len   Length of the range
         *
         * @return     A reference to the @ref sul::dynamic_bitset *this
         *
         * @pre        @code
         *             (pos < size()) && ((len == 0) || (pos + len - 1 < size()))
         *             @endcode
         *
         * @complexity Linear in @p len.
         *
         * @since      1.0.0
         */
        constexpr dynamic_bitset<Block, Allocator>& reset(size_type pos, size_type len);

        /**
         * @brief      Reset the bit at the position @p pos to @a false.
         *
         * @param[in]  pos    Position of the bit to reset
         *
         * @return     A reference to the @ref sul::dynamic_bitset *this
         *
         * @pre        @code
         *             pos < size()
         *             @endcode
         *
         * @complexity Constant.
         *
         * @since      1.0.0
         */
        constexpr dynamic_bitset<Block, Allocator>& reset(size_type pos);

        /**
         * @brief      Reset all the bits of the @ref sul::dynamic_bitset to @a false.
         *
         * @return     A reference to the @ref sul::dynamic_bitset *this
         *
         * @complexity Linear in the size of the @ref sul::dynamic_bitset.
         *
         * @since      1.0.0
         */
        constexpr dynamic_bitset<Block, Allocator>& reset();

        /**
         * @brief      Flip the bits of the range \[@p pos, @p pos + @p len\[.
         *
         * @param[in]  pos   Position of the first bit of the range
         * @param[in]  len   Length of the range
         *
         * @return     A reference to the @ref sul::dynamic_bitset *this
         *
         * @pre        @code
         *             (pos < size()) && ((len == 0) || (pos + len - 1 < size()))
         *             @endcode
         *
         * @complexity Linear in @p len.
         *
         * @since      1.0.0
         */
        constexpr dynamic_bitset<Block, Allocator>& flip(size_type pos, size_type len);

        /**
         * @brief      Flip the bit at the position @p pos.
         *
         * @param[in]  pos    Position of the bit to reset
         *
         * @return     A reference to the @ref sul::dynamic_bitset *this
         *
         * @pre        @code
         *             pos < size()
         *             @endcode
         *
         * @complexity Constant.
         *
         * @since      1.0.0
         */
        constexpr dynamic_bitset<Block, Allocator>& flip(size_type pos);

        /**
         * @brief      Flip all the bits of the @ref sul::dynamic_bitset.
         *
         * @return     A reference to the @ref sul::dynamic_bitset *this
         *
         * @complexity Linear in the size of the @ref sul::dynamic_bitset.
         *
         * @since      1.0.0
         */
        constexpr dynamic_bitset<Block, Allocator>& flip();

        /**
         * @brief      Test the value of the bit at position @p pos.
         *
         * @param[in]  pos   Position of the bit to test
         *
         * @return     The tested bit value
         *
         * @pre        @code
         *             pos < size()
         *             @endcode
         *
         * @complexity Constant.
         *
         * @since      1.0.0
         */
        [[nodiscard]] constexpr bool test(size_type pos) const;

        /**
         * @brief      Test the value of the bit at position @p pos and set it to @a true or value @p
         *             value.
         *
         * @param[in]  pos    Position of the bit to test and set
         * @param[in]  value  Value to set the bit to
         *
         * @return     The tested bit value
         *
         * @pre        @code
         *             pos < size()
         *             @endcode
         *
         * @complexity Constant.
         *
         * @since      1.0.0
         */
        [[nodiscard]] constexpr bool test_set(size_type pos, bool value = true);

        /**
         * @brief      Checks if all bits are set to @a true.
         *
         * @return     @a true if all bits are set to @a true, otherwise @a false
         *
         * @remark     Return @a true if the @ref sul::dynamic_bitset is empty, the logic is that you
         *             are checking if all bits are set to @a true, meaning none of them is set to @a
         *             false, and in an empty @ref sul::dynamic_bitset no bits are set to @a false.
         *
         * @complexity Linear in the size of the @ref sul::dynamic_bitset.
         *
         * @since      1.0.0
         */
        [[nodiscard]] constexpr bool all() const;

        /**
         * @brief      Checks if any bits are set to @a true.
         *
         * @return     @a true if any of the bits is set to @a true, otherwise @a false
         *
         * @remark     Return @a false if the @ref sul::dynamic_bitset is empty, the logic is you are
         *             checking if there is at least one bit set to @a true and in an empty @ref
         *             dynamic_bitset there is no bit set to @a true.
         *
         * @complexity Linear in the size of the @ref sul::dynamic_bitset.
         *
         * @since      1.0.0
         */
        [[nodiscard]] constexpr bool any() const;

        /**
         * @brief      Checks if none of the bits are set to @a true.
         *
         * @return     @a true if none of the bits is set to @a true, otherwise @a false
         *
         * @remark     Return @a true if the @ref sul::dynamic_bitset is empty, the logic is that you
         *             are checking if there is no bit set to @a true and in an empty @ref
         *             sul::dynamic_bitset there is no bit that can be set to @a true.
         *
         * @complexity Linear in the size of the @ref sul::dynamic_bitset.
         *
         * @since      1.0.0
         */
        [[nodiscard]] constexpr bool none() const;

        /**
         * @brief      Count the number of bits set to @a true.
         *
         * @details    Return 0 if the @ref sul::dynamic_bitset is empty.
         *
         * @return     The number of bits that are set to @a true
         *
         * @complexity Linear in the size of the @ref sul::dynamic_bitset.
         *
         * @since      1.0.0
         */
        [[nodiscard]] constexpr size_type count() const noexcept;

        /**
         * @brief      Accesses the bit at position @p pos.
         *
         * @param[in]  pos   Position of the bit to access
         *
         * @return     A @ref reference object which allows writing to the requested bit
         *
         * @pre        @code
         *             pos < size()
         *             @endcode
         *
         * @complexity Constant.
         *
         * @since      1.0.0
         */
        [[nodiscard]] constexpr reference operator[](size_type pos);

        /**
         * @brief      Accesses the bit at position @p pos.
         *
         * @param[in]  pos   Position of the bit to access
         *
         * @return     The value of the requested bit
         *
         * @pre        @code
         *             pos < size()
         *             @endcode
         *
         * @complexity Constant.
         *
         * @since      1.0.0
         */
        [[nodiscard]] constexpr const_reference operator[](size_type pos) const;

        /**
         * @brief      Give the number of bits of the @ref sul::dynamic_bitset.
         *
         * @return     The number of bits of the @ref sul::dynamic_bitset
         *
         * @complexity Constant.
         *
         * @since      1.0.0
         */
        [[nodiscard]] constexpr size_type size() const noexcept;

        /**
         * @brief      Give the number of blocks used by the @ref sul::dynamic_bitset.
         *
         * @return     The number of blocks used by the @ref sul::dynamic_bitset
         *
         * @complexity Constant.
         *
         * @since      1.0.0
         */
        [[nodiscard]] constexpr size_type num_blocks() const noexcept;

        /**
         * @brief      Checks if the @ref sul::dynamic_bitset is empty.
         *
         * @details    Equivalent to:
         *             @code
         *             size() == 0;
         *             @endcode
         *
         * @return     @a true if the @ref sul::dynamic_bitset is empty, @a false otherwise
         *
         * @complexity Constant.
         *
         * @since      1.0.0
         */
        [[nodiscard]] constexpr bool empty() const noexcept;

        /**
         * @brief      Give the number of bits that the @ref sul::dynamic_bitset has currently allocated
         *             space for.
         *
         * @return     Capacity of the currently allocated storage.
         *
         * @complexity Constant.
         *
         * @since      1.0.0
         */
        [[nodiscard]] constexpr size_type capacity() const noexcept;

        /**
         * @brief      Increase the capacity of the @ref sul::dynamic_bitset to a value that's greater
         *             or equal to @p num_bits.
         *
         * @details    If @p num_bits is greater than the current capacity, new storage is allocated and
         *             all @ref reference on bits of the @ref sul::dynamic_bitset are invalidated,
         *             otherwise the method does nothing.
         *
         * @param[in]  num_bits  New capacity of the @ref sul::dynamic_bitset
         *
         * @complexity At most linear in the size of the @ref sul::dynamic_bitset.
         *
         * @since      1.0.0
         */
        constexpr void reserve(size_type num_bits);

        /**
         * @brief      Requests the removal of unused capacity.
         *
         * @details    It is a non-binding request to reduce the capacity to the size. It depends on the
         *             implementation of std\::vector whether the request is fulfilled.\n If
         *             reallocation occurs, all @ref reference on bits of the @ref sul::dynamic_bitset
         *             are invalidated.
         *
         * @complexity At most linear in the size of the @ref sul::dynamic_bitset.
         *
         * @since      1.0.0
         */
        constexpr void shrink_to_fit();

        /**
         * @brief      Determines if this @ref sul::dynamic_bitset is a subset of @p bitset.
         *
         * @details    This @ref sul::dynamic_bitset is a subset of @p bitset if, for every bit that is
         *             set in this @ref sul::dynamic_bitset, the corresponding bit in @p bitset a is
         *             also set.\n\n Less efficient but equivalent way to get this result:
         *             @code
         *             res = (this & ~bitset).none();
         *             @endcode
         *
         * @param[in]  bitset  The @ref sul::dynamic_bitset for which to check if this @ref
         *                     sul::dynamic_bitset is a subset
         *
         * @return     @a true if this @ref sul::dynamic_bitset is a subset of @p bitset, @a false
         *             otherwise
         *
         * @remark     The relation "is a subset of" is not symmetric (A being a subset of B doesn't
         *             imply that B is a subset of A) but is antisymmetric (if A is a subset of B and B
         *             is a subset of A, then A == B).
         *
         * @pre        @code
         *             size() == bitset.size()
         *             @endcode
         *
         * @complexity Linear in the size of the @ref sul::dynamic_bitset.
         *
         * @since      1.0.0
         */
        [[nodiscard]] constexpr bool is_subset_of(const dynamic_bitset<Block, Allocator>& bitset) const;

        /**
         * @brief      Determines if this @ref sul::dynamic_bitset is a proper subset of @p bitset.
         *
         * @details    This @ref sul::dynamic_bitset is a proper subset of @p bitset if, for every bit
         *             that is set in this @ref sul::dynamic_bitset, the corresponding bit in @p bitset
         *             a is also set and if this @ref sul::dynamic_bitset is different from @p
         *             bitset.\n\n Less efficient but equivalent way to get this result:
         *             @code
         *             res = ((this != bitset) && (this & ~bitset).none());
         *             @endcode
         *
         * @param[in]  bitset  The @ref sul::dynamic_bitset for which to check if this @ref
         *                     sul::dynamic_bitset is a proper subset
         *
         * @return     @a true if this @ref sul::dynamic_bitset is a proper subset of @p bitset, @a
         *             false otherwise
         *
         * @remark     The relation "is a proper subset of" is asymmetric (A being a proper subset of B
         *             imply that B is not a proper subset of A).
         *
         * @pre        @code
         *             size() == bitset.size()
         *             @endcode
         *
         * @complexity Linear in the size of the @ref sul::dynamic_bitset.
         *
         * @since      1.0.0
         */
        [[nodiscard]] constexpr bool is_proper_subset_of(
            const dynamic_bitset<Block, Allocator>& bitset) const;

        /**
         * @brief      Determines if this @ref sul::dynamic_bitset and @p bitset intersect.
         *
         * @details    This @ref sul::dynamic_bitset intersects with @p bitset if for at least one bit
         *             set in this @ref sul::dynamic_bitset, the corresponding bit in @p bitset a is
         *             also set. In other words two bitsets intersect if they have at least one bit set
         *             in common.\n\n Less efficient but equivalent way to get this result:
         *             @code
         *             res = (this & bitset).any();
         *             @endcode
         *
         * @param[in]  bitset  The @ref sul::dynamic_bitset for which to check if this @ref
         *                     sul::dynamic_bitset intersects
         *
         * @return     @a true if this @ref sul::dynamic_bitset intersects with @p bitset, @a false
         *             otherwise
         *
         * @pre        @code
         *             size() == bitset.size()
         *             @endcode
         *
         * @complexity Linear in the size of the @ref sul::dynamic_bitset.
         *
         * @since      1.0.0
         */
        [[nodiscard]] constexpr bool intersects(const dynamic_bitset<Block, Allocator>& bitset) const;

        /**
         * @brief      Find the position of the first bit set in the @ref sul::dynamic_bitset starting
         *             from the least-significant bit.
         *
         * @details    Give the lowest index of the @ref sul::dynamic_bitset with a bit set, or @ref
         *             npos if no bits are set.
         *
         * @return     The position of the first bit set, or @ref npos if no bits are set
         *
         * @complexity Linear in the size of the @ref sul::dynamic_bitset.
         *
         * @since      1.0.0
         */
        [[nodiscard]] constexpr size_type find_first() const;

        /**
         * @brief      Find the position of the first bit set in the range \[@p prev + 1, @ref size()\[
         *             of the @ref sul::dynamic_bitset starting from the position @p prev + 1.
         *
         * @details    Give the lowest index superior to @p prev of the @ref sul::dynamic_bitset with a
         *             bit set, or @ref npos if no bits are set after the index @p prev.\n If @p prev +
         *             1 \>= @ref size(), return @ref npos.
         *
         * @param[in]  prev  Position of the bit preceding the search range
         *
         * @return     The position of the first bit set after @p prev, or @ref npos if no bits are set
         *             after @p prev
         *
         * @complexity Linear in @ref size() - @p prev.
         *
         * @since      1.0.0
         */
        [[nodiscard]] constexpr size_type find_next(size_type prev) const;

        /**
         * @brief      Exchanges the bits of this @ref sul::dynamic_bitset with those of @p other.
         *
         * @details    All @ref reference on bits of the @ref sul::dynamic_bitset are invalidated.
         *
         * @param      other  @ref sul::dynamic_bitset to exchange bits with
         *
         * @complexity Constant.
         *
         * @since      1.0.0
         */
        constexpr void swap(dynamic_bitset<Block, Allocator>& other);

        /**
         * @brief      Gets the associated allocator.
         *
         * @return     The associated allocator.
         *
         * @complexity Constant.
         *
         * @since      1.0.0
         */
        [[nodiscard]] constexpr allocator_type get_allocator() const;

        /**
         * @brief      Generate a string representation of the @ref sul::dynamic_bitset.
         *
         * @details    Uses @p zero to represent bits with value of @a false and @p one to represent
         *             bits with value of @a true. The resulting string contains @ref size() characters
         *             with the first character corresponds to the last (@ref size() - 1th) bit and the
         *             last character corresponding to the first bit.
         *
         * @param[in]  zero     Character to use to represent @a false
         * @param[in]  one      Character to use to represent @a true
         *
         * @tparam     _CharT   Character type of the string
         * @tparam     _Traits  Traits class specifying the operations on the character type of the
         *                      string
         * @tparam     _Alloc   Allocator type used to allocate internal storage of the string
         *
         * @return     The string representing the @ref sul::dynamic_bitset content
         *
         * @complexity Linear in the size of the @ref sul::dynamic_bitset.
         *
         * @since      1.0.0
         */
        template<typename _CharT = char,
            typename _Traits = std::char_traits<_CharT>,
            typename _Alloc = std::allocator<_CharT>>
            [[nodiscard]] constexpr std::basic_string<_CharT, _Traits, _Alloc> to_string(
                _CharT zero = _CharT('0'),
                _CharT one = _CharT('1')) const;

        /**
         * @brief      Converts the contents of the bitset to an <tt>unsigned long</tt> integer.
         *
         * @details    The first bit of the bitset corresponds to the least significant digit of the
         *             number and the last bit corresponds to the most significant digit.
         *
         * @return     The numeric value corresponding to the bitset contents.
         *
         * @throws     std::overflow_error  if the value is too large to be represented in an <tt>unsigned
         *                                  long</tt>
         *
         * @complexity Constant.
         *
         * @since      1.3.0
         */
        [[nodiscard]] constexpr unsigned long to_ulong() const;

        /**
         * @brief      Converts the contents of the bitset to an <tt>unsigned long long</tt> integer.
         *
         * @details    The first bit of the bitset corresponds to the least significant digit of the
         *             number and the last bit corresponds to the most significant digit.
         *
         * @return     The numeric value corresponding to the bitset contents.
         *
         * @throws     std::overflow_error  if the value is too large to be represented in an <tt>unsigned long
         *                                  long</tt>
         *
         * @complexity Constant.
         *
         * @since      1.3.0
         */
        [[nodiscard]] constexpr unsigned long long to_ullong() const;

        /**
         * @brief      Iterate on the @ref sul::dynamic_bitset and call @p function with the position of
         *             the bits on.
         *
         * @details    For each set bit, @p function is called as follow:
         *             @code
         *             std::invoke(std::forward<Function>(function), bit_pos, std::forward<Parameters>(parameters)...))
         *             @endcode
         *             where @p bit_pos is the position of the current bit on. Thus @p function
         *             should take a size_t for the current set bit position as first argument, also @p
         *             parameters can be used to pass additional arguments to @p function when it is
         *             called by this method.\n\n @p function can return nothing or a bool, if it return
         *             a bool, the return value indicate if the iteration should continue, @a true to
         *             continue the iteration, @a false to stop, this make it easy to do an early exit.
         *
         * @param      function    Function to call on all bits on, take the current bit position as
         *                         first argument and @p parameters as next arguments
         * @param      parameters  Extra parameters for @p function
         *
         * @tparam     Function    Type of @p function, must take a size_t as first argument and @p
         *                         Parameters as next arguments
         * @tparam     Parameters  Type of @p parameters
         *
         * @complexity Linear in the size of the @ref sul::dynamic_bitset.
         *
         * @since      1.0.0
         */
        template<typename Function, typename... Parameters>
        constexpr void iterate_bits_on(Function&& function, Parameters&&... parameters) const;

        /**
         * @brief      Return a pointer to the underlying array serving as blocks storage.
         *
         * @details    The pointer is such that range [@ref data(); @ref data() + @ref num_blocks()) is
         *             always a valid range, even if the container is empty (@ref data() is not
         *             dereferenceable in that case).
         *
         *
         * @post       The bits past the end of the @ref sul::dynamic_bitset in the last block are
         *             guaranteed to be 0s, example:
         *             @code
         *             // random bitset of size 11
         *             std::minstd_rand rand(std::random_device{}());
         *             std::bernoulli_distribution dist;
         *             sul::dynamic_bitset<uint8_t> bitset;
         *             for(size_t i = 0; i < 11; ++i)
         *             {
         *                 bitset.push_back(dist(rand));
         *             }
         *
         *             // the bitset use 2 blocks of 8 bits
         *             // check that unused bits are set to 0
         *             assert(*(bitset.data() + 1) >> 3 == 0);
         *             @endcode
         *
         * @remark     If the @ref sul::dynamic_bitset is empty, this function may or may not return a
         *             null pointer.
         *
         * @return     A pointer to the underlying array serving as blocks storage
         *
         * @complexity Constant.
         *
         * @since      1.2.0
         */
        [[nodiscard]] constexpr block_type* data() noexcept;

        /**
         * @brief      Return a pointer to the underlying array serving as blocks storage.
         *
         * @details    The pointer is such that range [@ref data(); @ref data() + @ref num_blocks()) is
         *             always a valid range, even if the container is empty (@ref data() is not
         *             dereferenceable in that case).
         *
         *
         * @post       The bits past the end of the @ref sul::dynamic_bitset in the last block are
         *             guaranteed to be 0s, example:
         *             @code
         *             // random bitset of size 11
         *             std::minstd_rand rand(std::random_device{}());
         *             std::bernoulli_distribution dist;
         *             sul::dynamic_bitset<uint8_t> bitset;
         *             for(size_t i = 0; i < 11; ++i)
         *             {
         *                 bitset.push_back(dist(rand));
         *             }
         *
         *             // the bitset use 2 blocks of 8 bits
         *             // check that unused bits are set to 0
         *             assert(*(bitset.data() + 1) >> 3 == 0);
         *             @endcode
         *
         * @remark     If the @ref sul::dynamic_bitset is empty, this function may or may not return a
         *             null pointer.
         *
         * @return     A pointer to the underlying array serving as blocks storage
         *
         * @complexity Constant.
         *
         * @since      1.2.0
         */
        [[nodiscard]] constexpr const block_type* data() const noexcept;

        /**
         * @brief      Test if two @ref sul::dynamic_bitset have the same content.
         *
         * @param[in]  lhs         The left hand side @ref sul::dynamic_bitset of the operator
         * @param[in]  rhs         The right hand side @ref sul::dynamic_bitset of the operator
         *
         * @tparam     Block_      Block type used by @p lhs and @p rhs for storing the bits
         * @tparam     Allocator_  Allocator type used by @p lhs and @p rhs for memory management
         *
         * @return     @a true if they contain the same bits, @a false otherwise
         *
         * @complexity Linear in the size of the @ref sul::dynamic_bitset.
         *
         * @since      1.0.0
         */
        template<typename Block_, typename Allocator_>
        friend constexpr bool operator==(const dynamic_bitset<Block_, Allocator_>& lhs,
            const dynamic_bitset<Block_, Allocator_>& rhs);

        /**
         * @brief      Test if @p lhs is "less than" @p rhs. The comparison of the two @ref
         *             dynamic_bitset is first on numbers their content represent and then on their
         *             size.
         *
         * @details    The size comparison is necessary for the comparison operators to keep their
         *             properties. For example without the size comparison the "<=" operator (defined
         *             for "A <= B" by "!(B < A)") would no longer be antisymmetric (if A \<= B and B
         *             \<= A, then A == B) because @ref operator==() compare the @ref
         *             sul::dynamic_bitset as a container and not a number. For example with bitsets
         *             A(0011) and B(011), without the size comparison B \< A would be @a false, A \<= B
         *             would be @a true, B \<= A would be @a true, but A == B would be @a false,
         *             breaking the antisymmetric property of the operator. Thus, to respect the
         *             properties of the operators, the size is used as a secondary criteria for the
         *             comparison of @ref sul::dynamic_bitset which content represent the same number.
         *             Therefore, for the previous example with bitsets A(0011) and B(011), B \< A is @a
         *             true, A \<= B is @a false, B \<= A is @a true and A == B is @a false.\n\n If
         *             comparing bitsets @a A and @a B with the content of @a A representing the number
         *             @a a, and the content of @a B representing the number @a b, this operator would
         *             work as follow:
         *             @code
         *             if(a == b)
         *             {
         *                 return A.size() < B.size();
         *             }
         *             else
         *             {
         *                 return a < b;
         *             }
         *             @endcode
         *
         * @remark     The empty @ref sul::dynamic_bitset is the "lowest" of all bitset and for 0-only
         *             bitsets comparison, the shortest is the lowest.
         *
         * @param[in]  lhs         The left hand side @ref sul::dynamic_bitset of the operator
         * @param[in]  rhs         The right hand side @ref sul::dynamic_bitset of the operator
         *
         * @tparam     Block_      Block type used by @p lhs and @p rhs for storing the bits
         * @tparam     Allocator_  Allocator type used by @p lhs and @p rhs for memory management
         *
         * @return     @a true if @p lhs is "less than" @p rhs
         *
         * @complexity Linear in the size of the @ref sul::dynamic_bitset.
         *
         * @since      1.0.0
         */
        template<typename Block_, typename Allocator_>
        friend constexpr bool operator<(const dynamic_bitset<Block_, Allocator_>& lhs,
            const dynamic_bitset<Block_, Allocator_>& rhs);

    private:
        template<typename T>
        struct dependent_false : public std::false_type
        {};

        std::vector<Block, Allocator> m_blocks;
        size_type m_bits_number;

        static constexpr block_type zero_block = block_type(0);
        static constexpr block_type one_block = block_type(~zero_block);
        static constexpr size_type block_last_bit_index = bits_per_block - 1;

        static constexpr size_type blocks_required(size_type nbits) noexcept;

        static constexpr size_type block_index(size_type pos) noexcept;
        static constexpr size_type bit_index(size_type pos) noexcept;

        static constexpr block_type bit_mask(size_type pos) noexcept;
        static constexpr block_type bit_mask(size_type first, size_type last) noexcept;

        static constexpr void set_block_bits(block_type& block,
            size_type first,
            size_type last,
            bool val = true) noexcept;
        static constexpr void flip_block_bits(block_type& block,
            size_type first,
            size_type last) noexcept;

        static constexpr size_type block_count(const block_type& block) noexcept;
        static constexpr size_type block_count(const block_type& block, size_type nbits) noexcept;

        static constexpr size_type count_block_trailing_zero(const block_type& block) noexcept;

        template<typename _CharT, typename _Traits>
        constexpr void init_from_string(std::basic_string_view<_CharT, _Traits> str,
            typename std::basic_string_view<_CharT, _Traits>::size_type pos,
            typename std::basic_string_view<_CharT, _Traits>::size_type n,
            _CharT zero,
            _CharT one);

        constexpr block_type& get_block(size_type pos);
        constexpr const block_type& get_block(size_type pos) const;
        constexpr block_type& last_block();
        constexpr block_type last_block() const;

        // used bits in the last block
        constexpr size_type extra_bits_number() const noexcept;
        // unused bits in the last block
        constexpr size_type unused_bits_number() const noexcept;

        template<typename BinaryOperation>
        constexpr void apply(const dynamic_bitset<Block, Allocator>& other, BinaryOperation binary_op);
        template<typename UnaryOperation>
        constexpr void apply(UnaryOperation unary_op);
        constexpr void apply_left_shift(size_type shift);
        constexpr void apply_right_shift(size_type shift);

        // reset unused bits to 0
        constexpr void sanitize();

        // check functions used in asserts
        constexpr bool check_unused_bits() const noexcept;
        constexpr bool check_size() const noexcept;
        constexpr bool check_consistency() const noexcept;
    };

    // Deduction guideline for expressions like "dynamic_bitset a(32);" with an integral type as parameter
    // to use the constructor with the initial size instead of the constructor with the allocator.
    template<typename integral_type, typename = std::enable_if_t<std::is_integral_v<integral_type>>>
    dynamic_bitset(integral_type) -> dynamic_bitset<>;

    //=================================================================================================
    // dynamic_bitset external functions declarations
    //=================================================================================================

    /**
     * @brief      Test if two @ref sul::dynamic_bitset content are different.
     *
     * @details    Defined as:
     *             @code
     *             return !(lhs == rhs);
     *             @endcode
     *             see @ref sul::dynamic_bitset::operator==() for more informations.
     *
     * @param[in]  lhs        The left hand side @ref sul::dynamic_bitset of the operator
     * @param[in]  rhs        The right hand side @ref sul::dynamic_bitset of the operator
     *
     * @tparam     Block      Block type used by @p lhs and @p rhs for storing the bits
     * @tparam     Allocator  Allocator type used by @p lhs and @p rhs for memory management
     *
     * @return     @a true if they does not contain the same bits, @a false otherwise
     *
     * @complexity Linear in the size of the @ref sul::dynamic_bitset.
     *
     * @since      1.0.0
     *
     * @relatesalso dynamic_bitset
     */
    template<typename Block, typename Allocator>
    constexpr bool operator!=(const dynamic_bitset<Block, Allocator>& lhs,
        const dynamic_bitset<Block, Allocator>& rhs);

    /**
     * @brief      Test if @p lhs is "less than or equal to" @p rhs. The comparison of the two @ref
     *             sul::dynamic_bitset is first on numbers their content represent and then on their size.
     *
     * @details    Defined as:
     *             @code
     *             return !(rhs < lhs);
     *             @endcode
     *             see @ref sul::dynamic_bitset::operator<() for more informations.
     *
     * @param[in]  lhs        The left hand side @ref sul::dynamic_bitset of the operator
     * @param[in]  rhs        The right hand side @ref sul::dynamic_bitset of the operator
     *
     * @tparam     Block      Block type used by @p lhs and @p rhs for storing the bits
     * @tparam     Allocator  Allocator type used by @p lhs and @p rhs for memory management
     *
     * @return     @a true if @p lhs is "less than or equal to" @p rhs
     *
     * @complexity Linear in the size of the @ref sul::dynamic_bitset.
     *
     * @since      1.0.0
     *
     * @relatesalso dynamic_bitset
     */
    template<typename Block, typename Allocator>
    constexpr bool operator<=(const dynamic_bitset<Block, Allocator>& lhs,
        const dynamic_bitset<Block, Allocator>& rhs);

    /**
     * @brief      Test if @p lhs is "greater than" @p rhs. The comparison of the two @ref
     *             sul::dynamic_bitset is first on numbers their content represent and then on their
     *             size.
     *
     * @details    Defined as:
     *             @code
     *             return rhs < lhs;
     *             @endcode
     *             see @ref sul::dynamic_bitset::operator<() for more informations.
     *
     * @param[in]  lhs        The left hand side @ref sul::dynamic_bitset of the operator
     * @param[in]  rhs        The right hand side @ref sul::dynamic_bitset of the operator
     *
     * @tparam     Block      Block type used by @p lhs and @p rhs for storing the bits
     * @tparam     Allocator  Allocator type used by @p lhs and @p rhs for memory management
     *
     * @return     @a true if @p lhs is "greater than" @p rhs
     *
     * @complexity Linear in the size of the @ref sul::dynamic_bitset.
     *
     * @since      1.0.0
     *
     * @relatesalso dynamic_bitset
     */
    template<typename Block, typename Allocator>
    constexpr bool operator>(const dynamic_bitset<Block, Allocator>& lhs,
        const dynamic_bitset<Block, Allocator>& rhs);

    /**
     * @brief      Test if @p lhs is "greater than or equal to" @p rhs. The comparison of the two @ref
     *             sul::dynamic_bitset is first on numbers their content represent and then on their
     *             size.
     *
     * @details    Defined as:
     *             @code
     *             return !(lhs < rhs);
     *             @endcode
     *             see @ref sul::dynamic_bitset::operator<() for more informations.
     *
     * @param[in]  lhs        The left hand side @ref sul::dynamic_bitset of the operator
     * @param[in]  rhs        The right hand side @ref sul::dynamic_bitset of the operator
     *
     * @tparam     Block      Block type used by @p lhs and @p rhs for storing the bits
     * @tparam     Allocator  Allocator type used by @p lhs and @p rhs for memory management
     *
     * @return     @a true if @p lhs is "greater than or equal to" @p rhs
     *
     * @complexity Linear in the size of the @ref sul::dynamic_bitset.
     *
     * @since      1.0.0
     *
     * @relatesalso dynamic_bitset
     */
    template<typename Block, typename Allocator>
    constexpr bool operator>=(const dynamic_bitset<Block, Allocator>& lhs,
        const dynamic_bitset<Block, Allocator>& rhs);

    /**
     * @brief      Performs binary AND on corresponding pairs of bits of @p lhs and @p rhs.
     *
     * @details    Defined as:
     *             @code
     *             dynamic_bitset<Block, Allocator> result(lhs);
     *             return result &= rhs;
     *             @endcode
     *             see @ref sul::dynamic_bitset::operator&=() for more informations.
     *
     * @param[in]  lhs        The left hand side @ref sul::dynamic_bitset of the operator
     * @param[in]  rhs        The right hand side @ref sul::dynamic_bitset of the operator
     *
     * @tparam     Block      Block type used by @p lhs and @p rhs for storing the bits
     * @tparam     Allocator  Allocator type used by @p lhs and @p rhs for memory management
     *
     * @return     A @ref sul::dynamic_bitset with each bit being the result of a binary AND between the
     *             corresponding pair of bits of @p lhs and @p rhs
     *
     * @pre        @code
     *             lhs.size() == rhs.size()
     *             @endcode
     *
     * @complexity Linear in the size of the @ref sul::dynamic_bitset.
     *
     * @since      1.0.0
     *
     * @relatesalso dynamic_bitset
     */
    template<typename Block, typename Allocator>
    constexpr dynamic_bitset<Block, Allocator> operator&(const dynamic_bitset<Block, Allocator>& lhs,
        const dynamic_bitset<Block, Allocator>& rhs);

    /**
     * @brief      Performs binary OR on corresponding pairs of bits of @p lhs and @p rhs.
     *
     * @details    Defined as:
     *             @code
     *             dynamic_bitset<Block, Allocator> result(lhs);
     *             return result |= rhs;
     *             @endcode
     *             see @ref sul::dynamic_bitset::operator|=() for more informations.
     *
     * @param[in]  lhs        The left hand side @ref sul::dynamic_bitset of the operator
     * @param[in]  rhs        The right hand side @ref sul::dynamic_bitset of the operator
     *
     * @tparam     Block      Block type used by @p lhs and @p rhs for storing the bits
     * @tparam     Allocator  Allocator type used by @p lhs and @p rhs for memory management
     *
     * @return     A @ref sul::dynamic_bitset with each bit being the result of a binary OR between the
     *             corresponding pair of bits of @p lhs and @p rhs
     *
     * @pre        @code
     *             lhs.size() == rhs.size()
     *             @endcode
     *
     * @complexity Linear in the size of the @ref sul::dynamic_bitset.
     *
     * @since      1.0.0
     *
     * @relatesalso dynamic_bitset
     */
    template<typename Block, typename Allocator>
    constexpr dynamic_bitset<Block, Allocator> operator|(const dynamic_bitset<Block, Allocator>& lhs,
        const dynamic_bitset<Block, Allocator>& rhs);

    /**
     * @brief      Performs binary XOR on corresponding pairs of bits of @p lhs and @p rhs.
     *
     * @details    Defined as:
     *             @code
     *             dynamic_bitset<Block, Allocator> result(lhs);
     *             return result ^= rhs;
     *             @endcode
     *             see @ref sul::dynamic_bitset::operator^=() for more informations.
     *
     * @param[in]  lhs        The left hand side @ref sul::dynamic_bitset of the operator
     * @param[in]  rhs        The right hand side @ref sul::dynamic_bitset of the operator
     *
     * @tparam     Block      Block type used by @p lhs and @p rhs for storing the bits
     * @tparam     Allocator  Allocator type used by @p lhs and @p rhs for memory management
     *
     * @return     A @ref sul::dynamic_bitset with each bit being the result of a binary XOR between the
     *             corresponding pair of bits of @p lhs and @p rhs
     *
     * @pre        @code
     *             lhs.size() == rhs.size()
     *             @endcode
     *
     * @complexity Linear in the size of the @ref sul::dynamic_bitset.
     *
     * @since      1.0.0
     *
     * @relatesalso dynamic_bitset
     */
    template<typename Block, typename Allocator>
    constexpr dynamic_bitset<Block, Allocator> operator^(const dynamic_bitset<Block, Allocator>& lhs,
        const dynamic_bitset<Block, Allocator>& rhs);

    /**
     * @brief      Performs binary difference between bits of @p lhs and @p rhs.
     *
     * @details    Defined as:
     *             @code
     *             dynamic_bitset<Block, Allocator> result(lhs);
     *             return result -= rhs;
     *             @endcode
     *             see @ref sul::dynamic_bitset::operator-=() for more informations.
     *
     * @param[in]  lhs        The left hand side @ref sul::dynamic_bitset of the operator
     * @param[in]  rhs        The right hand side @ref sul::dynamic_bitset of the operator
     *
     * @tparam     Block      Block type used by @p lhs and @p rhs for storing the bits
     * @tparam     Allocator  Allocator type used by @p lhs and @p rhs for memory management
     *
     * @return     A @ref sul::dynamic_bitset with each bit being the result of the binary difference
     *             between the corresponding bits of @p lhs and @p rhs
     *
     * @pre        @code
     *             lhs.size() == rhs.size()
     *             @endcode
     *
     * @complexity Linear in the size of the @ref sul::dynamic_bitset.
     *
     * @since      1.0.0
     *
     * @relatesalso dynamic_bitset
     */
    template<typename Block, typename Allocator>
    constexpr dynamic_bitset<Block, Allocator> operator-(const dynamic_bitset<Block, Allocator>& lhs,
        const dynamic_bitset<Block, Allocator>& rhs);

    /**
     * @brief      Insert a string representation of this @ref sul::dynamic_bitset to a character
     *             stream.
     *
     * @details    The string representation written is the same as if generated with @ref
     *             sul::dynamic_bitset::to_string() with default parameter, using '1' for @a true bits
     *             and '0' for @a false bits.
     *
     * @param      os         Character stream to write to
     * @param[in]  bitset     @ref sul::dynamic_bitset to write
     *
     * @tparam     _CharT     Character type of the character stream
     * @tparam     _Traits    Traits class specifying the operations on the character type of the
     *                        character stream
     * @tparam     Block      Block type used by @p bitset for storing the bits
     * @tparam     Allocator  Allocator type used by @p bitset for memory management
     *
     * @return     @p os
     *
     * @complexity Linear in the size of the @ref sul::dynamic_bitset.
     *
     * @since      1.0.0
     *
     * @relatesalso dynamic_bitset
     */
    template<typename _CharT, typename _Traits, typename Block, typename Allocator>
    constexpr std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& os,
        const dynamic_bitset<Block, Allocator>& bitset);

    /**
     * @brief      Extract a @ref sul::dynamic_bitset from a character stream using its string
     *             representation.
     *
     * @details    The string representation expected is the same as if generated with @ref
     *             sul::dynamic_bitset::to_string() with default parameter, using '1' for @a true bits
     *             and '0' for @a false bits. On success the content of @p bitset is cleared before
     *             writing to it. The extraction starts by skipping leading whitespace then take the
     *             characters one by one and stop if @p is.good() return @a false or the next character
     *             is neither _CharT('0') nor _CharT('1').
     *
     * @param      is         Character stream to read from
     * @param      bitset     @ref sul::dynamic_bitset to write to
     *
     * @tparam     _CharT     Character type of the character stream
     * @tparam     _Traits    Traits class specifying the operations on the character type of the
     *                        character stream
     * @tparam     Block      Block type used by @p bitset for storing the bits
     * @tparam     Allocator  Allocator type used by @p bitset for memory management
     *
     * @return     @p is
     *
     * @complexity Linear in the size of the @ref sul::dynamic_bitset.
     *
     * @since      1.0.0
     *
     * @relatesalso dynamic_bitset
     */
    template<typename _CharT, typename _Traits, typename Block, typename Allocator>
    constexpr std::basic_istream<_CharT, _Traits>& operator>>(std::basic_istream<_CharT, _Traits>& is,
        dynamic_bitset<Block, Allocator>& bitset);

    /**
     * @brief      Exchange the content of @p bitset1 and @p bitset2.
     *
     * @details    Defined as:
     *             @code
     *             bitset1.swap(bitset2);
     *             @endcode
     *             see @ref sul::dynamic_bitset::swap() for more informations.
     *
     * @param      bitset1    @ref sul::dynamic_bitset to be swapped
     * @param      bitset2    @ref sul::dynamic_bitset to be swapped
     *
     * @tparam     Block      Block type used by @p bitset for storing the bits
     * @tparam     Allocator  Allocator type used by @p bitset for memory management
     *
     * @complexity Constant.
     *
     * @since      1.0.0
     *
     * @relatesalso dynamic_bitset
     */
    template<typename Block, typename Allocator>
    constexpr void swap(dynamic_bitset<Block, Allocator>& bitset1,
        dynamic_bitset<Block, Allocator>& bitset2);

    //=================================================================================================
    // dynamic_bitset::reference functions implementations
    //=================================================================================================

    template<typename Block, typename Allocator>
    constexpr dynamic_bitset<Block, Allocator>::reference::reference(
        dynamic_bitset<Block, Allocator>& bitset,
        size_type bit_pos)
        : m_block(bitset.get_block(bit_pos)), m_mask(dynamic_bitset<Block, Allocator>::bit_mask(bit_pos))
    {}

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::reference& dynamic_bitset<Block, Allocator>::
        reference::operator=(bool v)
    {
        assign(v);
        return *this;
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::reference& dynamic_bitset<Block, Allocator>::
        reference::operator=(const dynamic_bitset<Block, Allocator>::reference& rhs)
    {
        assign(rhs);
        return *this;
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::reference& dynamic_bitset<Block, Allocator>::
        reference::operator=(dynamic_bitset::reference&& rhs) noexcept
    {
        assign(rhs);
        return *this;
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::reference& dynamic_bitset<Block, Allocator>::
        reference::operator&=(bool v)
    {
        if (!v)
        {
            reset();
        }
        return *this;
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::reference& dynamic_bitset<Block, Allocator>::
        reference::operator|=(bool v)
    {
        if (v)
        {
            set();
        }
        return *this;
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::reference& dynamic_bitset<Block, Allocator>::
        reference::operator^=(bool v)
    {
        if (v)
        {
            flip();
        }
        return *this;
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::reference& dynamic_bitset<Block, Allocator>::
        reference::operator-=(bool v)
    {
        if (v)
        {
            reset();
        }
        return *this;
    }

    template<typename Block, typename Allocator>
    constexpr bool dynamic_bitset<Block, Allocator>::reference::operator~() const
    {
        return (m_block & m_mask) == zero_block;
    }

    template<typename Block, typename Allocator>
    constexpr dynamic_bitset<Block, Allocator>::reference::operator bool() const
    {
        return (m_block & m_mask) != zero_block;
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::reference& dynamic_bitset<Block, Allocator>::
        reference::set()
    {
        m_block |= m_mask;
        return *this;
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::reference& dynamic_bitset<Block, Allocator>::
        reference::reset()
    {
        m_block &= static_cast<block_type>(~m_mask);
        return *this;
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::reference& dynamic_bitset<Block, Allocator>::
        reference::flip()
    {
        m_block ^= m_mask;
        return *this;
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::reference& dynamic_bitset<Block, Allocator>::
        reference::assign(bool v)
    {
        if (v)
        {
            set();
        }
        else
        {
            reset();
        }
        return *this;
    }

    //=================================================================================================
    // dynamic_bitset public functions implementations
    //=================================================================================================

    template<typename Block, typename Allocator>
    constexpr dynamic_bitset<Block, Allocator>::dynamic_bitset(const allocator_type& allocator)
        : m_blocks(allocator), m_bits_number(0)
    {}

    template<typename Block, typename Allocator>
    constexpr dynamic_bitset<Block, Allocator>::dynamic_bitset(size_type nbits,
        unsigned long long init_val,
        const allocator_type& allocator)
        : m_blocks(blocks_required(nbits), allocator), m_bits_number(nbits)
    {
        if (nbits == 0 || init_val == 0)
        {
            return;
        }

        constexpr size_t ull_bits_number = std::numeric_limits<unsigned long long>::digits;
        constexpr size_t init_val_required_blocks = ull_bits_number / bits_per_block;
        if constexpr (init_val_required_blocks == 1)
        {
            m_blocks[0] = init_val;
        }
        else
        {
            const unsigned long long block_mask = static_cast<unsigned long long>(one_block);
            const size_t blocks_to_init = std::min(m_blocks.size(), init_val_required_blocks);
            for (size_t i = 0; i < blocks_to_init; ++i)
            {
                m_blocks[i] = block_type((init_val >> (i * bits_per_block) & block_mask));
            }
        }
        sanitize();
    }

    template<typename Block, typename Allocator>
    constexpr dynamic_bitset<Block, Allocator>::dynamic_bitset(
        std::initializer_list<block_type> init_vals,
        const allocator_type& allocator)
        : m_blocks(allocator), m_bits_number(0)
    {
        append(init_vals);
    }

    template<typename Block, typename Allocator>
    template<typename _CharT, typename _Traits>
    constexpr dynamic_bitset<Block, Allocator>::dynamic_bitset(
        std::basic_string_view<_CharT, _Traits> str,
        typename std::basic_string_view<_CharT, _Traits>::size_type pos,
        typename std::basic_string_view<_CharT, _Traits>::size_type n,
        _CharT zero,
        _CharT one,
        const allocator_type& allocator)
        : m_blocks(allocator), m_bits_number(0)
    {
        assert(pos < str.size());
        init_from_string(str, pos, n, zero, one);
    }

    template<typename Block, typename Allocator>
    template<typename _CharT, typename _Traits, typename _Alloc>
    constexpr dynamic_bitset<Block, Allocator>::dynamic_bitset(
        const std::basic_string<_CharT, _Traits, _Alloc>& str,
        typename std::basic_string<_CharT, _Traits, _Alloc>::size_type pos,
        typename std::basic_string<_CharT, _Traits, _Alloc>::size_type n,
        _CharT zero,
        _CharT one,
        const allocator_type& allocator)
        : m_blocks(allocator), m_bits_number(0)
    {
        assert(pos < str.size());
        init_from_string(std::basic_string_view<_CharT, _Traits>(str), pos, n, zero, one);
    }

    template<typename Block, typename Allocator>
    template<typename _CharT, typename _Traits>
    constexpr dynamic_bitset<Block, Allocator>::dynamic_bitset(
        const _CharT* str,
        typename std::basic_string<_CharT>::size_type pos,
        typename std::basic_string<_CharT>::size_type n,
        _CharT zero,
        _CharT one,
        const allocator_type& allocator)
        : m_blocks(allocator), m_bits_number(0)
    {
        init_from_string(std::basic_string_view<_CharT, _Traits>(str), pos, n, zero, one);
    }

    template<typename Block, typename Allocator>
    constexpr void dynamic_bitset<Block, Allocator>::resize(size_type nbits, bool value)
    {
        if (nbits == m_bits_number)
        {
            return;
        }

        const size_type old_num_blocks = num_blocks();
        const size_type new_num_blocks = blocks_required(nbits);

        const block_type init_value = value ? one_block : zero_block;
        if (new_num_blocks != old_num_blocks)
        {
            m_blocks.resize(new_num_blocks, init_value);
        }

        if (value && nbits > m_bits_number && old_num_blocks > 0)
        {
            // set value of the new bits in the old last block
            const size_type extra_bits = extra_bits_number();
            if (extra_bits > 0)
            {
                m_blocks[old_num_blocks - 1] |= static_cast<block_type>(init_value << extra_bits);
            }
        }

        m_bits_number = nbits;
        sanitize();
        assert(check_consistency());
    }

    template<typename Block, typename Allocator>
    constexpr void dynamic_bitset<Block, Allocator>::clear()
    {
        m_blocks.clear();
        m_bits_number = 0;
    }

    template<typename Block, typename Allocator>
    constexpr void dynamic_bitset<Block, Allocator>::push_back(bool value)
    {
        const size_type new_last_bit = m_bits_number++;
        if (m_bits_number <= m_blocks.size() * bits_per_block)
        {
            if (value)
            {
                set(new_last_bit, value);
            }
        }
        else
        {
            m_blocks.push_back(block_type(value));
        }
        assert(operator[](new_last_bit) == value);
        assert(check_consistency());
    }

    template<typename Block, typename Allocator>
    constexpr void dynamic_bitset<Block, Allocator>::pop_back()
    {
        if (empty())
        {
            return;
        }

        --m_bits_number;
        if (m_blocks.size() > blocks_required(m_bits_number))
        {
            m_blocks.pop_back();
            // no extra bits: sanitize not required
            assert(extra_bits_number() == 0);
        }
        else
        {
            sanitize();
        }
        assert(check_consistency());
    }

    template<typename Block, typename Allocator>
    constexpr void dynamic_bitset<Block, Allocator>::append(block_type block)
    {
        const size_type extra_bits = extra_bits_number();
        if (extra_bits == 0)
        {
            m_blocks.push_back(block);
        }
        else
        {
            last_block() |= static_cast<block_type>(block << extra_bits);
            m_blocks.push_back(block_type(block >> (bits_per_block - extra_bits)));
        }

        m_bits_number += bits_per_block;
        assert(check_consistency());
    }

    template<typename Block, typename Allocator>
    constexpr void dynamic_bitset<Block, Allocator>::append(std::initializer_list<block_type> blocks)
    {
        if (blocks.size() == 0)
        {
            return;
        }

        append(std::cbegin(blocks), std::cend(blocks));
    }

    template<typename Block, typename Allocator>
    template<typename BlockInputIterator>
    constexpr void dynamic_bitset<Block, Allocator>::append(BlockInputIterator first,
        BlockInputIterator last)
    {
        if (first == last)
        {
            return;
        }

        // if random access iterators, std::distance complexity is constant
        if constexpr (std::is_same_v<
            typename std::iterator_traits<BlockInputIterator>::iterator_category,
            std::random_access_iterator_tag>)
        {
            assert(std::distance(first, last) > 0);
            m_blocks.reserve(m_blocks.size() + static_cast<size_type>(std::distance(first, last)));
        }

        const size_type extra_bits = extra_bits_number();
        const size_type unused_bits = unused_bits_number();
        if (extra_bits == 0)
        {
            auto pos = m_blocks.insert(std::end(m_blocks), first, last);
            assert(std::distance(pos, std::end(m_blocks)) > 0);
            m_bits_number +=
                static_cast<size_type>(std::distance(pos, std::end(m_blocks))) * bits_per_block;
        }
        else
        {
            last_block() |= static_cast<block_type>(*first << extra_bits);
            block_type block = block_type(*first >> unused_bits);
            ++first;
            while (first != last)
            {
                block |= static_cast<block_type>(*first << extra_bits);
                m_blocks.push_back(block);
                m_bits_number += bits_per_block;
                block = block_type(*first >> unused_bits);
                ++first;
            }
            m_blocks.push_back(block);
            m_bits_number += bits_per_block;
        }

        assert(check_consistency());
    }
    template<typename Block, typename Allocator>
    constexpr dynamic_bitset<Block, Allocator>& dynamic_bitset<Block, Allocator>::operator&=(
        const dynamic_bitset<Block, Allocator>& rhs)
    {
        assert(size() == rhs.size());
        //apply(rhs, std::bit_and());
        for (size_type i = 0; i < m_blocks.size(); ++i)
        {
            m_blocks[i] &= rhs.m_blocks[i];
        }
        return *this;
    }

    template<typename Block, typename Allocator>
    constexpr dynamic_bitset<Block, Allocator>& dynamic_bitset<Block, Allocator>::operator|=(
        const dynamic_bitset<Block, Allocator>& rhs)
    {
        assert(size() == rhs.size());
        //apply(rhs, std::bit_or());
        for (size_type i = 0; i < m_blocks.size(); ++i)
        {
            m_blocks[i] |= rhs.m_blocks[i];
        }
        return *this;
    }

    template<typename Block, typename Allocator>
    constexpr dynamic_bitset<Block, Allocator>& dynamic_bitset<Block, Allocator>::operator^=(
        const dynamic_bitset<Block, Allocator>& rhs)
    {
        assert(size() == rhs.size());
        //apply(rhs, std::bit_xor());
        for (size_type i = 0; i < m_blocks.size(); ++i)
        {
            m_blocks[i] ^= rhs.m_blocks[i];
        }
        return *this;
    }

    template<typename Block, typename Allocator>
    constexpr dynamic_bitset<Block, Allocator>& dynamic_bitset<Block, Allocator>::operator-=(
        const dynamic_bitset<Block, Allocator>& rhs)
    {
        assert(size() == rhs.size());
        //apply(rhs, [](const block_type& x, const block_type& y) { return (x & ~y); });
        for (size_type i = 0; i < m_blocks.size(); ++i)
        {
            m_blocks[i] &= static_cast<block_type>(~rhs.m_blocks[i]);
        }
        return *this;
    }

    template<typename Block, typename Allocator>
    constexpr dynamic_bitset<Block, Allocator>& dynamic_bitset<Block, Allocator>::operator<<=(
        size_type shift)
    {
        if (shift != 0)
        {
            if (shift >= m_bits_number)
            {
                reset();
            }
            else
            {
                apply_left_shift(shift);
                sanitize(); // unused bits can have changed, reset them to 0
            }
        }
        return *this;
    }

    template<typename Block, typename Allocator>
    constexpr dynamic_bitset<Block, Allocator>& dynamic_bitset<Block, Allocator>::operator>>=(
        size_type shift)
    {
        if (shift != 0)
        {
            if (shift >= m_bits_number)
            {
                reset();
            }
            else
            {
                apply_right_shift(shift);
            }
        }
        return *this;
    }

    template<typename Block, typename Allocator>
    constexpr dynamic_bitset<Block, Allocator> dynamic_bitset<Block, Allocator>::operator<<(
        size_type shift) const
    {
        return dynamic_bitset<Block, Allocator>(*this) <<= shift;
    }

    template<typename Block, typename Allocator>
    constexpr dynamic_bitset<Block, Allocator> dynamic_bitset<Block, Allocator>::operator>>(
        size_type shift) const
    {
        return dynamic_bitset<Block, Allocator>(*this) >>= shift;
    }

    template<typename Block, typename Allocator>
    constexpr dynamic_bitset<Block, Allocator> dynamic_bitset<Block, Allocator>::operator~() const
    {
        dynamic_bitset<Block, Allocator> bitset(*this);
        bitset.flip();
        return bitset;
    }

    template<typename Block, typename Allocator>
    constexpr dynamic_bitset<Block, Allocator>& dynamic_bitset<Block, Allocator>::set(size_type pos,
        size_type len,
        bool value)
    {
        assert(pos < size());
        if (len == 0)
        {
            return *this;
        }
        assert(pos + len - 1 < size());

        const size_type first_block = block_index(pos);
        const size_type last_block = block_index(pos + len - 1);
        const size_type first_bit_index = bit_index(pos);
        const size_type last_bit_index = bit_index(pos + len - 1);

        if (first_block == last_block)
        {
            set_block_bits(m_blocks[first_block], first_bit_index, last_bit_index, value);
        }
        else
        {
            size_type first_full_block = first_block;
            size_type last_full_block = last_block;

            if (first_bit_index != 0)
            {
                ++first_full_block; // first block is not full
                set_block_bits(m_blocks[first_block], first_bit_index, block_last_bit_index, value);
            }

            if (last_bit_index != block_last_bit_index)
            {
                --last_full_block; // last block is not full
                set_block_bits(m_blocks[last_block], 0, last_bit_index, value);
            }

            const block_type full_block = value ? one_block : zero_block;
            for (size_type i = first_full_block; i <= last_full_block; ++i)
            {
                m_blocks[i] = full_block;
            }
        }

        return *this;
    }

    template<typename Block, typename Allocator>
    constexpr dynamic_bitset<Block, Allocator>& dynamic_bitset<Block, Allocator>::set(size_type pos,
        bool value)
    {
        assert(pos < size());

        if (value)
        {
            m_blocks[block_index(pos)] |= bit_mask(pos);
        }
        else
        {
            m_blocks[block_index(pos)] &= static_cast<block_type>(~bit_mask(pos));
        }

        return *this;
    }

    template<typename Block, typename Allocator>
    constexpr dynamic_bitset<Block, Allocator>& dynamic_bitset<Block, Allocator>::set()
    {
        std::fill(std::begin(m_blocks), std::end(m_blocks), one_block);
        sanitize();
        return *this;
    }

    template<typename Block, typename Allocator>
    constexpr dynamic_bitset<Block, Allocator>& dynamic_bitset<Block, Allocator>::reset(size_type pos,
        size_type len)
    {
        return set(pos, len, false);
    }

    template<typename Block, typename Allocator>
    constexpr dynamic_bitset<Block, Allocator>& dynamic_bitset<Block, Allocator>::reset(size_type pos)
    {
        return set(pos, false);
    }

    template<typename Block, typename Allocator>
    constexpr dynamic_bitset<Block, Allocator>& dynamic_bitset<Block, Allocator>::reset()
    {
        std::fill(std::begin(m_blocks), std::end(m_blocks), zero_block);
        return *this;
    }

    template<typename Block, typename Allocator>
    constexpr dynamic_bitset<Block, Allocator>& dynamic_bitset<Block, Allocator>::flip(size_type pos,
        size_type len)
    {
        assert(pos < size());
        if (len == 0)
        {
            return *this;
        }
        assert(pos + len - 1 < size());

        const size_type first_block = block_index(pos);
        const size_type last_block = block_index(pos + len - 1);
        const size_type first_bit_index = bit_index(pos);
        const size_type last_bit_index = bit_index(pos + len - 1);

        if (first_block == last_block)
        {
            flip_block_bits(m_blocks[first_block], first_bit_index, last_bit_index);
        }
        else
        {
            size_type first_full_block = first_block;
            size_type last_full_block = last_block;

            if (first_bit_index != 0)
            {
                ++first_full_block; // first block is not full
                flip_block_bits(m_blocks[first_block], first_bit_index, block_last_bit_index);
            }

            if (last_bit_index != block_last_bit_index)
            {
                --last_full_block; // last block is not full
                flip_block_bits(m_blocks[last_block], 0, last_bit_index);
            }

            for (size_type i = first_full_block; i <= last_full_block; ++i)
            {
                m_blocks[i] = block_type(~m_blocks[i]);
            }
        }

        return *this;
    }

    template<typename Block, typename Allocator>
    constexpr dynamic_bitset<Block, Allocator>& dynamic_bitset<Block, Allocator>::flip(size_type pos)
    {
        assert(pos < size());
        m_blocks[block_index(pos)] ^= bit_mask(pos);
        return *this;
    }

    template<typename Block, typename Allocator>
    constexpr dynamic_bitset<Block, Allocator>& dynamic_bitset<Block, Allocator>::flip()
    {
        std::transform(
            std::cbegin(m_blocks), std::cend(m_blocks), std::begin(m_blocks), std::bit_not<block_type>());
        sanitize();
        return *this;
    }

    template<typename Block, typename Allocator>
    constexpr bool dynamic_bitset<Block, Allocator>::test(size_type pos) const
    {
        assert(pos < size());
        return (m_blocks[block_index(pos)] & bit_mask(pos)) != zero_block;
    }

    template<typename Block, typename Allocator>
    constexpr bool dynamic_bitset<Block, Allocator>::test_set(size_type pos, bool value)
    {
        bool const result = test(pos);
        if (result != value)
        {
            set(pos, value);
        }
        return result;
    }

    template<typename Block, typename Allocator>
    constexpr bool dynamic_bitset<Block, Allocator>::all() const
    {
        if (empty())
        {
            return true;
        }

        const block_type full_block = one_block;
        if (extra_bits_number() == 0)
        {
            for (const block_type& block : m_blocks)
            {
                if (block != full_block)
                {
                    return false;
                }
            }
        }
        else
        {
            for (size_type i = 0; i < m_blocks.size() - 1; ++i)
            {
                if (m_blocks[i] != full_block)
                {
                    return false;
                }
            }
            if (last_block() != (full_block >> unused_bits_number()))
            {
                return false;
            }
        }
        return true;
    }

    template<typename Block, typename Allocator>
    constexpr bool dynamic_bitset<Block, Allocator>::any() const
    {
        for (const block_type& block : m_blocks)
        {
            if (block != zero_block)
            {
                return true;
            }
        }
        return false;
    }

    template<typename Block, typename Allocator>
    constexpr bool dynamic_bitset<Block, Allocator>::none() const
    {
        return !any();
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::size_type dynamic_bitset<Block,
        Allocator>::count()
        const noexcept
    {
        if (empty())
        {
            return 0;
        }

#if DYNAMIC_BITSET_CAN_USE_LIBPOPCNT
        const size_type count =
            static_cast<size_type>(popcnt(m_blocks.data(), m_blocks.size() * sizeof(block_type)));
#else
        size_type count = 0;

        // full blocks
        for (size_type i = 0; i < m_blocks.size() - 1; ++i)
        {
            count += block_count(m_blocks[i]);
        }

        // last block
        const block_type& block = last_block();
        const size_type extra_bits = extra_bits_number();
        if (extra_bits == 0)
        {
            count += block_count(block);
        }
        else
        {
            count += block_count(block, extra_bits);
        }
#endif
        return count;
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::reference dynamic_bitset<Block, Allocator>::
        operator[](size_type pos)
    {
        assert(pos < size());
        return dynamic_bitset<Block, Allocator>::reference(*this, pos);
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::const_reference dynamic_bitset<
        Block,
        Allocator>::operator[](size_type pos) const
    {
        return test(pos);
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::size_type dynamic_bitset<Block,
        Allocator>::size()
        const noexcept
    {
        return m_bits_number;
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::size_type dynamic_bitset<Block, Allocator>::
        num_blocks() const noexcept
    {
        return m_blocks.size();
    }

    template<typename Block, typename Allocator>
    constexpr bool dynamic_bitset<Block, Allocator>::empty() const noexcept
    {
        return size() == 0;
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::size_type dynamic_bitset<Block,
        Allocator>::capacity()
        const noexcept
    {
        return m_blocks.capacity() * bits_per_block;
    }

    template<typename Block, typename Allocator>
    constexpr void dynamic_bitset<Block, Allocator>::reserve(size_type num_bits)
    {
        m_blocks.reserve(blocks_required(num_bits));
    }

    template<typename Block, typename Allocator>
    constexpr void dynamic_bitset<Block, Allocator>::shrink_to_fit()
    {
        m_blocks.shrink_to_fit();
    }

    template<typename Block, typename Allocator>
    constexpr bool dynamic_bitset<Block, Allocator>::is_subset_of(
        const dynamic_bitset<Block, Allocator>& bitset) const
    {
        assert(size() == bitset.size());
        for (size_type i = 0; i < m_blocks.size(); ++i)
        {
            if ((m_blocks[i] & ~bitset.m_blocks[i]) != zero_block)
            {
                return false;
            }
        }
        return true;
    }

    template<typename Block, typename Allocator>
    constexpr bool dynamic_bitset<Block, Allocator>::is_proper_subset_of(
        const dynamic_bitset<Block, Allocator>& bitset) const
    {
        assert(size() == bitset.size());
        bool is_proper = false;
        for (size_type i = 0; i < m_blocks.size(); ++i)
        {
            const block_type& self_block = m_blocks[i];
            const block_type& other_block = bitset.m_blocks[i];

            if ((self_block & ~other_block) != zero_block)
            {
                return false;
            }
            if ((~self_block & other_block) != zero_block)
            {
                is_proper = true;
            }
        }
        return is_proper;
    }

    template<typename Block, typename Allocator>
    constexpr bool dynamic_bitset<Block, Allocator>::intersects(
        const dynamic_bitset<Block, Allocator>& bitset) const
    {
        const size_type min_blocks_number = std::min(m_blocks.size(), bitset.m_blocks.size());
        for (size_type i = 0; i < min_blocks_number; ++i)
        {
            if ((m_blocks[i] & bitset.m_blocks[i]) != zero_block)
            {
                return true;
            }
        }
        return false;
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::size_type dynamic_bitset<Block, Allocator>::
        find_first() const
    {
        for (size_type i = 0; i < m_blocks.size(); ++i)
        {
            if (m_blocks[i] != zero_block)
            {
                return i * bits_per_block + count_block_trailing_zero(m_blocks[i]);
            }
        }
        return npos;
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::size_type dynamic_bitset<Block, Allocator>::
        find_next(size_type prev) const
    {
        if (empty() || prev >= (size() - 1))
        {
            return npos;
        }

        const size_type first_bit = prev + 1;
        const size_type first_block = block_index(first_bit);
        const size_type first_bit_index = bit_index(first_bit);
        const block_type first_block_shifted = block_type(m_blocks[first_block] >> first_bit_index);

        if (first_block_shifted != zero_block)
        {
            return first_bit + count_block_trailing_zero(first_block_shifted);
        }
        else
        {
            for (size_type i = first_block + 1; i < m_blocks.size(); ++i)
            {
                if (m_blocks[i] != zero_block)
                {
                    return i * bits_per_block + count_block_trailing_zero(m_blocks[i]);
                }
            }
        }
        return npos;
    }

    template<typename Block, typename Allocator>
    constexpr void dynamic_bitset<Block, Allocator>::swap(dynamic_bitset<Block, Allocator>& other)
    {
        std::swap(m_blocks, other.m_blocks);
        std::swap(m_bits_number, other.m_bits_number);
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::allocator_type dynamic_bitset<
        Block,
        Allocator>::get_allocator() const
    {
        return m_blocks.get_allocator();
    }

    template<typename Block, typename Allocator>
    template<typename _CharT, typename _Traits, typename _Alloc>
    constexpr std::basic_string<_CharT, _Traits, _Alloc> dynamic_bitset<Block, Allocator>::to_string(
        _CharT zero,
        _CharT one) const
    {
        const size_type len = size();
        std::basic_string<_CharT, _Traits, _Alloc> str(len, zero);
        for (size_type i_block = 0; i_block < m_blocks.size(); ++i_block)
        {
            if (m_blocks[i_block] == zero_block)
            {
                continue;
            }
            block_type mask = block_type(1);
            const size_type limit =
                i_block * bits_per_block < len ? len - i_block * bits_per_block : bits_per_block;
            for (size_type i_bit = 0; i_bit < limit; ++i_bit)
            {
                if ((m_blocks[i_block] & mask) != zero_block)
                {
                    _Traits::assign(str[len - (i_block * bits_per_block + i_bit + 1)], one);
                }
                // mask <<= 1; not used because it trigger -Wconversion because of integral promotion for block_type smaller than int
                mask = static_cast<block_type>(mask << 1);
            }
        }
        return str;
    }

    template<typename Block, typename Allocator>
    constexpr unsigned long dynamic_bitset<Block, Allocator>::to_ulong() const
    {
        if (m_bits_number == 0)
        {
            return 0;
        }

        constexpr size_t ul_bits_number = std::numeric_limits<unsigned long>::digits;
        if (find_next(ul_bits_number - 1) != npos)
        {
            throw std::overflow_error("sul::dynamic_bitset::to_ulong");
        }

        unsigned long result = 0;
        const size_type result_bits_number = std::min(ul_bits_number, m_bits_number);
        for (size_type i_block = 0; i_block <= block_index(result_bits_number - 1); ++i_block)
        {
            result |= (static_cast<unsigned long>(m_blocks[i_block]) << (i_block * bits_per_block));
        }

        return result;
    }

    template<typename Block, typename Allocator>
    constexpr unsigned long long dynamic_bitset<Block, Allocator>::to_ullong() const
    {
        if (m_bits_number == 0)
        {
            return 0;
        }

        constexpr size_t ull_bits_number = std::numeric_limits<unsigned long long>::digits;
        if (find_next(ull_bits_number - 1) != npos)
        {
            throw std::overflow_error("sul::dynamic_bitset::to_ullong");
        }

        unsigned long long result = 0;
        const size_type result_bits_number = std::min(ull_bits_number, m_bits_number);
        for (size_type i_block = 0; i_block <= block_index(result_bits_number - 1); ++i_block)
        {
            result |=
                (static_cast<unsigned long long>(m_blocks[i_block]) << (i_block * bits_per_block));
        }

        return result;
    }

    template<typename Block, typename Allocator>
    template<typename Function, typename... Parameters>
    constexpr void dynamic_bitset<Block, Allocator>::iterate_bits_on(Function&& function,
        Parameters&&... parameters) const
    {
        if constexpr (!std::is_invocable_v<Function, size_t, Parameters...>)
        {
            static_assert(dependent_false<Function>::value, "Function take invalid arguments");
            // function should take (size_t, parameters...) as arguments
        }

        if constexpr (std::is_same_v<std::invoke_result_t<Function, size_t, Parameters...>, void>)
        {
            size_type i_bit = find_first();
            while (i_bit != npos)
            {
                std::invoke(
                    std::forward<Function>(function), i_bit, std::forward<Parameters>(parameters)...);
                i_bit = find_next(i_bit);
            }
        }
        else if constexpr (std::is_convertible_v<std::invoke_result_t<Function, size_t, Parameters...>,
            bool>)
        {
            size_type i_bit = find_first();
            while (i_bit != npos)
            {
                if (!std::invoke(
                    std::forward<Function>(function), i_bit, std::forward<Parameters>(parameters)...))
                {
                    break;
                }
                i_bit = find_next(i_bit);
            }
        }
        else
        {
            static_assert(dependent_false<Function>::value, "Function have invalid return type");
            // return type should be void, or convertible to bool
        }
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::block_type* dynamic_bitset<Block, Allocator>::
        data() noexcept
    {
        return m_blocks.data();
    }

    template<typename Block, typename Allocator>
    constexpr const typename dynamic_bitset<Block, Allocator>::block_type* dynamic_bitset<
        Block,
        Allocator>::data() const noexcept
    {
        return m_blocks.data();
    }

    template<typename Block_, typename Allocator_>
    [[nodiscard]] constexpr bool operator==(const dynamic_bitset<Block_, Allocator_>& lhs,
        const dynamic_bitset<Block_, Allocator_>& rhs)
    {
        return (lhs.m_bits_number == rhs.m_bits_number) && (lhs.m_blocks == rhs.m_blocks);
    }

    template<typename Block_, typename Allocator_>
    [[nodiscard]] constexpr bool operator<(const dynamic_bitset<Block_, Allocator_>& lhs,
        const dynamic_bitset<Block_, Allocator_>& rhs)
    {
        using size_type = typename dynamic_bitset<Block_, Allocator_>::size_type;
        using block_type = typename dynamic_bitset<Block_, Allocator_>::block_type;
        const size_type lhs_size = lhs.size();
        const size_type rhs_size = rhs.size();
        const size_type lhs_blocks_size = lhs.m_blocks.size();
        const size_type rhs_blocks_size = rhs.m_blocks.size();

        if (lhs_size == rhs_size)
        {
            // if comparison of two empty bitsets
            if (lhs_size == 0)
            {
                return false;
            }

            for (size_type i = lhs_blocks_size - 1; i > 0; --i)
            {
                if (lhs.m_blocks[i] != rhs.m_blocks[i])
                {
                    return lhs.m_blocks[i] < rhs.m_blocks[i];
                }
            }
            return lhs.m_blocks[0] < rhs.m_blocks[0];
        }

        // empty bitset inferior to 0-only bitset
        if (lhs_size == 0)
        {
            return true;
        }
        if (rhs_size == 0)
        {
            return false;
        }

        const bool rhs_longer = rhs_size > lhs_size;
        const dynamic_bitset<Block_, Allocator_>& longest_bitset = rhs_longer ? rhs : lhs;
        const size_type longest_blocks_size = std::max(lhs_blocks_size, rhs_blocks_size);
        const size_type shortest_blocks_size = std::min(lhs_blocks_size, rhs_blocks_size);
        for (size_type i = longest_blocks_size - 1; i >= shortest_blocks_size; --i)
        {
            if (longest_bitset.m_blocks[i] != block_type(0))
            {
                return rhs_longer;
            }
        }

        for (size_type i = shortest_blocks_size - 1; i > 0; --i)
        {
            if (lhs.m_blocks[i] != rhs.m_blocks[i])
            {
                return lhs.m_blocks[i] < rhs.m_blocks[i];
            }
        }
        if (lhs.m_blocks[0] != rhs.m_blocks[0])
        {
            return lhs.m_blocks[0] < rhs.m_blocks[0];
        }
        return lhs_size < rhs_size;
    }

    //=================================================================================================
    // dynamic_bitset private functions implementations
    //=================================================================================================

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::size_type dynamic_bitset<Block, Allocator>::
        blocks_required(size_type nbits) noexcept
    {
        return nbits / bits_per_block + static_cast<size_type>(nbits % bits_per_block > 0);
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::size_type dynamic_bitset<Block, Allocator>::
        block_index(size_type pos) noexcept
    {
        return pos / bits_per_block;
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::size_type dynamic_bitset<Block, Allocator>::
        bit_index(size_type pos) noexcept
    {
        return pos % bits_per_block;
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::block_type dynamic_bitset<Block, Allocator>::
        bit_mask(size_type pos) noexcept
    {
        return block_type(block_type(1) << bit_index(pos));
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::block_type dynamic_bitset<Block, Allocator>::
        bit_mask(size_type first, size_type last) noexcept
    {
        first = bit_index(first);
        last = bit_index(last);
        if (last == (block_last_bit_index))
        {
            return block_type(one_block << first);
        }
        else
        {
            return block_type(((block_type(1) << (last + 1)) - 1) ^ ((block_type(1) << first) - 1));
        }
    }

    template<typename Block, typename Allocator>
    constexpr void dynamic_bitset<Block, Allocator>::set_block_bits(block_type& block,
        size_type first,
        size_type last,
        bool val) noexcept
    {
        if (val)
        {
            block |= bit_mask(first, last);
        }
        else
        {
            block &= static_cast<block_type>(~bit_mask(first, last));
        }
    }

    template<typename Block, typename Allocator>
    constexpr void dynamic_bitset<Block, Allocator>::flip_block_bits(block_type& block,
        size_type first,
        size_type last) noexcept
    {
        block ^= bit_mask(first, last);
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::size_type dynamic_bitset<Block, Allocator>::
        block_count(const block_type& block) noexcept
    {
#if DYNAMIC_BITSET_CAN_USE_STD_BITOPS
        return static_cast<size_type>(std::popcount(block));
#else
        if (block == zero_block)
        {
            return 0;
        }

#	if DYNAMIC_BITSET_CAN_USE_GCC_BUILTIN || DYNAMIC_BITSET_CAN_USE_CLANG_BUILTIN_POPCOUNT
        constexpr size_t u_bits_number = std::numeric_limits<unsigned>::digits;
        constexpr size_t ul_bits_number = std::numeric_limits<unsigned long>::digits;
        constexpr size_t ull_bits_number = std::numeric_limits<unsigned long long>::digits;
        if constexpr (bits_per_block <= u_bits_number)
        {
            return static_cast<size_type>(__builtin_popcount(static_cast<unsigned int>(block)));
        }
        else if constexpr (bits_per_block <= ul_bits_number)
        {
            return static_cast<size_type>(__builtin_popcountl(static_cast<unsigned long>(block)));
        }
        else if constexpr (bits_per_block <= ull_bits_number)
        {
            return static_cast<size_type>(__builtin_popcountll(static_cast<unsigned long long>(block)));
        }
#	endif

        size_type count = 0;
        block_type mask = 1;
        for (size_type bit_index = 0; bit_index < bits_per_block; ++bit_index)
        {
            count += static_cast<size_type>((block & mask) != zero_block);
            // mask <<= 1; not used because it trigger -Wconversion because of integral promotion for block_type smaller than int
            mask = static_cast<block_type>(mask << 1);
        }
        return count;
#endif
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::size_type dynamic_bitset<Block, Allocator>::
        block_count(const block_type& block, size_type nbits) noexcept
    {
        assert(nbits <= bits_per_block);
#if DYNAMIC_BITSET_CAN_USE_STD_BITOPS
        const block_type shifted_block = block_type(block << (bits_per_block - nbits));
        return static_cast<size_type>(std::popcount(shifted_block));
#else
        const block_type shifted_block = block_type(block << (bits_per_block - nbits));
        if (shifted_block == zero_block)
        {
            return 0;
        }

#	if DYNAMIC_BITSET_CAN_USE_GCC_BUILTIN || DYNAMIC_BITSET_CAN_USE_CLANG_BUILTIN_POPCOUNT
        constexpr size_t u_bits_number = std::numeric_limits<unsigned>::digits;
        constexpr size_t ul_bits_number = std::numeric_limits<unsigned long>::digits;
        constexpr size_t ull_bits_number = std::numeric_limits<unsigned long long>::digits;
        if constexpr (bits_per_block <= u_bits_number)
        {
            return static_cast<size_type>(__builtin_popcount(static_cast<unsigned int>(shifted_block)));
        }
        else if constexpr (bits_per_block <= ul_bits_number)
        {
            return static_cast<size_type>(
                __builtin_popcountl(static_cast<unsigned long>(shifted_block)));
        }
        else if constexpr (bits_per_block <= ull_bits_number)
        {
            return static_cast<size_type>(
                __builtin_popcountll(static_cast<unsigned long long>(shifted_block)));
        }
#	endif

        size_type count = 0;
        block_type mask = 1;
        for (size_type bit_index = 0; bit_index < nbits; ++bit_index)
        {
            count += static_cast<size_type>((block & mask) != zero_block);
            // mask <<= 1; not used because it trigger -Wconversion because of integral promotion for block_type smaller than int
            mask = static_cast<block_type>(mask << 1);
        }

        return count;
#endif
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::size_type dynamic_bitset<Block, Allocator>::
        count_block_trailing_zero(const block_type& block) noexcept
    {
        assert(block != zero_block);
#if DYNAMIC_BITSET_CAN_USE_STD_BITOPS
        return static_cast<size_type>(std::countr_zero(block));
#else
#	if DYNAMIC_BITSET_CAN_USE_GCC_BUILTIN || DYNAMIC_BITSET_CAN_USE_CLANG_BUILTIN_CTZ
        constexpr size_t u_bits_number = std::numeric_limits<unsigned>::digits;
        constexpr size_t ul_bits_number = std::numeric_limits<unsigned long>::digits;
        constexpr size_t ull_bits_number = std::numeric_limits<unsigned long long>::digits;
        if constexpr (bits_per_block <= u_bits_number)
        {
            return static_cast<size_type>(__builtin_ctz(static_cast<unsigned int>(block)));
        }
        else if constexpr (bits_per_block <= ul_bits_number)
        {
            return static_cast<size_type>(__builtin_ctzl(static_cast<unsigned long>(block)));
        }
        else if constexpr (bits_per_block <= ull_bits_number)
        {
            return static_cast<size_type>(__builtin_ctzll(static_cast<unsigned long long>(block)));
        }

#	elif DYNAMIC_BITSET_CAN_USE_MSVC_BUILTIN_BITSCANFORWARD
        constexpr size_t ul_bits_number = std::numeric_limits<unsigned long>::digits;
        constexpr size_t ui64_bits_number = std::numeric_limits<unsigned __int64>::digits;
        if constexpr (bits_per_block <= ul_bits_number)
        {
            unsigned long index = std::numeric_limits<unsigned long>::max();
            _BitScanForward(&index, static_cast<unsigned long>(block));
            return static_cast<size_type>(index);
        }
        else if constexpr (bits_per_block <= ui64_bits_number)
        {
#		if DYNAMIC_BITSET_CAN_USE_MSVC_BUILTIN_BITSCANFORWARD64
            unsigned long index = std::numeric_limits<unsigned long>::max();
            _BitScanForward64(&index, static_cast<unsigned __int64>(block));
            return static_cast<size_type>(index);
#		else
            constexpr unsigned long max_ul = std::numeric_limits<unsigned long>::max();
            unsigned long low = block & max_ul;
            if (low != 0)
            {
                unsigned long index = std::numeric_limits<unsigned long>::max();
                _BitScanForward(&index, low);
                return static_cast<size_type>(index);
            }
            unsigned long high = block >> ul_bits_number;
            unsigned long index = std::numeric_limits<unsigned long>::max();
            _BitScanForward(&index, high);
            return static_cast<size_type>(ul_bits_number + index);
#		endif
        }
#	endif

        block_type mask = block_type(1);
        for (size_type i = 0; i < bits_per_block; ++i)
        {
            if ((block & mask) != zero_block)
            {
                return i;
            }
            // mask <<= 1; not used because it trigger -Wconversion because of integral promotion for block_type smaller than int
            mask = static_cast<block_type>(mask << 1);
        }
        assert(false); // LCOV_EXCL_LINE: unreachable
        return npos; // LCOV_EXCL_LINE: unreachable
#endif
    }

    template<typename Block, typename Allocator>
    template<typename _CharT, typename _Traits>
    constexpr void dynamic_bitset<Block, Allocator>::init_from_string(
        std::basic_string_view<_CharT, _Traits> str,
        typename std::basic_string_view<_CharT, _Traits>::size_type pos,
        typename std::basic_string_view<_CharT, _Traits>::size_type n,
        [[maybe_unused]] _CharT zero,
        _CharT one)
    {
        assert(pos < str.size());

        const size_type size = std::min(n, str.size() - pos);
        m_bits_number = size;

        m_blocks.clear();
        m_blocks.resize(blocks_required(size));
        for (size_t i = 0; i < size; ++i)
        {
            const _CharT c = str[(pos + size - 1) - i];
            assert(c == zero || c == one);
            if (c == one)
            {
                set(i);
            }
        }
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::block_type& dynamic_bitset<Block, Allocator>::
        get_block(size_type pos)
    {
        return m_blocks[block_index(pos)];
    }

    template<typename Block, typename Allocator>
    constexpr const typename dynamic_bitset<Block, Allocator>::block_type& dynamic_bitset<
        Block,
        Allocator>::get_block(size_type pos) const
    {
        return m_blocks[block_index(pos)];
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::block_type& dynamic_bitset<Block, Allocator>::
        last_block()
    {
        return m_blocks[m_blocks.size() - 1];
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::block_type dynamic_bitset<Block, Allocator>::
        last_block() const
    {
        return m_blocks[m_blocks.size() - 1];
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::size_type dynamic_bitset<Block, Allocator>::
        extra_bits_number() const noexcept
    {
        return bit_index(m_bits_number);
    }

    template<typename Block, typename Allocator>
    constexpr typename dynamic_bitset<Block, Allocator>::size_type dynamic_bitset<Block, Allocator>::
        unused_bits_number() const noexcept
    {
        return bits_per_block - extra_bits_number();
    }

    template<typename Block, typename Allocator>
    template<typename BinaryOperation>
    constexpr void dynamic_bitset<Block, Allocator>::apply(
        const dynamic_bitset<Block, Allocator>& other,
        BinaryOperation binary_op)
    {
        assert(num_blocks() == other.num_blocks());
        std::transform(std::cbegin(m_blocks),
            std::cend(m_blocks),
            std::cbegin(other.m_blocks),
            std::begin(m_blocks),
            binary_op);
    }

    template<typename Block, typename Allocator>
    template<typename UnaryOperation>
    constexpr void dynamic_bitset<Block, Allocator>::apply(UnaryOperation unary_op)
    {
        std::transform(std::cbegin(m_blocks), std::cend(m_blocks), std::begin(m_blocks), unary_op);
    }

    template<typename Block, typename Allocator>
    constexpr void dynamic_bitset<Block, Allocator>::apply_left_shift(size_type shift)
    {
        assert(shift > 0);
        assert(shift < capacity());

        const size_type blocks_shift = shift / bits_per_block;
        const size_type bits_offset = shift % bits_per_block;

        if (bits_offset == 0)
        {
            for (size_type i = m_blocks.size() - 1; i >= blocks_shift; --i)
            {
                m_blocks[i] = m_blocks[i - blocks_shift];
            }
        }
        else
        {
            const size_type reverse_bits_offset = bits_per_block - bits_offset;
            for (size_type i = m_blocks.size() - 1; i > blocks_shift; --i)
            {
                m_blocks[i] =
                    block_type((m_blocks[i - blocks_shift] << bits_offset)
                        | block_type(m_blocks[i - blocks_shift - 1] >> reverse_bits_offset));
            }
            m_blocks[blocks_shift] = block_type(m_blocks[0] << bits_offset);
        }

        // set bit that came at the right to 0 in unmodified blocks
        std::fill(std::begin(m_blocks),
            std::begin(m_blocks)
            + static_cast<typename decltype(m_blocks)::difference_type>(blocks_shift),
            zero_block);
    }

    template<typename Block, typename Allocator>
    constexpr void dynamic_bitset<Block, Allocator>::apply_right_shift(size_type shift)
    {
        assert(shift > 0);
        assert(shift < capacity());

        const size_type blocks_shift = shift / bits_per_block;
        const size_type bits_offset = shift % bits_per_block;
        const size_type last_block_to_shift = m_blocks.size() - blocks_shift - 1;

        if (bits_offset == 0)
        {
            for (size_type i = 0; i <= last_block_to_shift; ++i)
            {
                m_blocks[i] = m_blocks[i + blocks_shift];
            }
        }
        else
        {
            const size_type reverse_bits_offset = bits_per_block - bits_offset;
            for (size_type i = 0; i < last_block_to_shift; ++i)
            {
                m_blocks[i] =
                    block_type((m_blocks[i + blocks_shift] >> bits_offset)
                        | block_type(m_blocks[i + blocks_shift + 1] << reverse_bits_offset));
            }
            m_blocks[last_block_to_shift] = block_type(m_blocks[m_blocks.size() - 1] >> bits_offset);
        }

        // set bit that came at the left to 0 in unmodified blocks
        std::fill(
            std::begin(m_blocks)
            + static_cast<typename decltype(m_blocks)::difference_type>(last_block_to_shift + 1),
            std::end(m_blocks),
            zero_block);
    }

    template<typename Block, typename Allocator>
    constexpr void dynamic_bitset<Block, Allocator>::sanitize()
    {
        size_type shift = m_bits_number % bits_per_block;
        if (shift > 0)
        {
            last_block() &= static_cast<block_type>(~(one_block << shift));
        }
    }

    template<typename Block, typename Allocator>
    constexpr bool dynamic_bitset<Block, Allocator>::check_unused_bits() const noexcept
    {
        const size_type extra_bits = extra_bits_number();
        if (extra_bits > 0)
        {
            return (last_block() & (one_block << extra_bits)) == zero_block;
        }
        return true;
    }

    template<typename Block, typename Allocator>
    constexpr bool dynamic_bitset<Block, Allocator>::check_size() const noexcept
    {
        return blocks_required(size()) == m_blocks.size();
    }

    template<typename Block, typename Allocator>
    constexpr bool dynamic_bitset<Block, Allocator>::check_consistency() const noexcept
    {
        return check_unused_bits() && check_size();
    }

    //=================================================================================================
    // dynamic_bitset external functions implementations
    //=================================================================================================

    template<typename Block, typename Allocator>
    constexpr bool operator!=(const dynamic_bitset<Block, Allocator>& lhs,
        const dynamic_bitset<Block, Allocator>& rhs)
    {
        return !(lhs == rhs);
    }

    template<typename Block, typename Allocator>
    constexpr bool operator<=(const dynamic_bitset<Block, Allocator>& lhs,
        const dynamic_bitset<Block, Allocator>& rhs)
    {
        return !(rhs < lhs);
    }

    template<typename Block, typename Allocator>
    constexpr bool operator>(const dynamic_bitset<Block, Allocator>& lhs,
        const dynamic_bitset<Block, Allocator>& rhs)
    {
        return rhs < lhs;
    }

    template<typename Block, typename Allocator>
    constexpr bool operator>=(const dynamic_bitset<Block, Allocator>& lhs,
        const dynamic_bitset<Block, Allocator>& rhs)
    {
        return !(lhs < rhs);
    }

    template<typename Block, typename Allocator>
    constexpr dynamic_bitset<Block, Allocator> operator&(const dynamic_bitset<Block, Allocator>& lhs,
        const dynamic_bitset<Block, Allocator>& rhs)
    {
        dynamic_bitset<Block, Allocator> result(lhs);
        return result &= rhs;
    }

    template<typename Block, typename Allocator>
    constexpr dynamic_bitset<Block, Allocator> operator|(const dynamic_bitset<Block, Allocator>& lhs,
        const dynamic_bitset<Block, Allocator>& rhs)
    {
        dynamic_bitset<Block, Allocator> result(lhs);
        return result |= rhs;
    }

    template<typename Block, typename Allocator>
    constexpr dynamic_bitset<Block, Allocator> operator^(const dynamic_bitset<Block, Allocator>& lhs,
        const dynamic_bitset<Block, Allocator>& rhs)
    {
        dynamic_bitset<Block, Allocator> result(lhs);
        return result ^= rhs;
    }

    template<typename Block, typename Allocator>
    constexpr dynamic_bitset<Block, Allocator> operator-(const dynamic_bitset<Block, Allocator>& lhs,
        const dynamic_bitset<Block, Allocator>& rhs)
    {
        dynamic_bitset<Block, Allocator> result(lhs);
        return result -= rhs;
    }

    template<typename _CharT, typename _Traits, typename Block, typename Allocator>
    constexpr std::basic_ostream<_CharT, _Traits>& operator<<(
        std::basic_ostream<_CharT, _Traits>& os,
        const dynamic_bitset<Block, Allocator>& bitset)
    {
        // A better implementation is possible
        return os << bitset.template to_string<_CharT, _Traits>();
    }

    template<typename _CharT, typename _Traits, typename Block, typename Allocator>
    constexpr std::basic_istream<_CharT, _Traits>& operator>>(std::basic_istream<_CharT, _Traits>& is,
        dynamic_bitset<Block, Allocator>& bitset)
    {
        // A better implementation is possible
        constexpr _CharT zero = _CharT('0');
        constexpr _CharT one = _CharT('1');
        typename std::basic_istream<_CharT, _Traits>::sentry s(is);
        if (!s)
        {
            return is;
        }

        dynamic_bitset<Block, Allocator> reverse_bitset;
        _CharT val;
        is.get(val);
        while (is.good())
        {
            if (val == one)
            {
                reverse_bitset.push_back(true);
            }
            else if (val == zero)
            {
                reverse_bitset.push_back(false);
            }
            else
            {
                is.unget();
                break;
            }
            is.get(val);
        }

        bitset.clear();
        if (!reverse_bitset.empty())
        {
            for (typename dynamic_bitset<Block, Allocator>::size_type i = reverse_bitset.size() - 1;
                i > 0;
                --i)
            {
                bitset.push_back(reverse_bitset.test(i));
            }
            bitset.push_back(reverse_bitset.test(0));
        }

        return is;
    }

    template<typename Block, typename Allocator>
    constexpr void swap(dynamic_bitset<Block, Allocator>& bitset1,
        dynamic_bitset<Block, Allocator>& bitset2)
    {
        bitset1.swap(bitset2);
    }

#ifndef DYNAMIC_BITSET_NO_NAMESPACE
} // namespace sul
#endif

#endif //SUL_DYNAMIC_BITSET_HPP

#pragma endregion

namespace bv
{

    // forward declarations
    class Allocator;
    class PhysicalDevice;
    class Context;
    class DebugMessenger;
    class Surface;
    class Queue;
    class Device;
    class Image;
    class Swapchain;
    class ImageView;
    class ShaderModule;
    class Sampler;
    class DescriptorSetLayout;
    class PipelineLayout;
    class RenderPass;
    class GraphicsPipeline;
    class ComputePipeline;
    class Framebuffer;
    class CommandBuffer;
    class CommandPool;
    class Semaphore;
    class Fence;
    class Buffer;
    class DeviceMemory;
    class DescriptorSet;
    class DescriptorPool;
    class BufferView;
    class PipelineCache;
    class MemoryRegion;
    class MemoryChunk;
    class MemoryBank;

    // smart pointer type aliases
    _BV_DEFINE_SMART_PTR_TYPE_ALIASES(Allocator);
    _BV_DEFINE_SMART_PTR_TYPE_ALIASES(Context);
    _BV_DEFINE_SMART_PTR_TYPE_ALIASES(DebugMessenger);
    _BV_DEFINE_SMART_PTR_TYPE_ALIASES(Surface);
    _BV_DEFINE_SMART_PTR_TYPE_ALIASES(Queue);
    _BV_DEFINE_SMART_PTR_TYPE_ALIASES(Device);
    _BV_DEFINE_SMART_PTR_TYPE_ALIASES(Image);
    _BV_DEFINE_SMART_PTR_TYPE_ALIASES(Swapchain);
    _BV_DEFINE_SMART_PTR_TYPE_ALIASES(ImageView);
    _BV_DEFINE_SMART_PTR_TYPE_ALIASES(ShaderModule);
    _BV_DEFINE_SMART_PTR_TYPE_ALIASES(Sampler);
    _BV_DEFINE_SMART_PTR_TYPE_ALIASES(DescriptorSetLayout);
    _BV_DEFINE_SMART_PTR_TYPE_ALIASES(PipelineLayout);
    _BV_DEFINE_SMART_PTR_TYPE_ALIASES(RenderPass);
    _BV_DEFINE_SMART_PTR_TYPE_ALIASES(GraphicsPipeline);
    _BV_DEFINE_SMART_PTR_TYPE_ALIASES(ComputePipeline);
    _BV_DEFINE_SMART_PTR_TYPE_ALIASES(Framebuffer);
    _BV_DEFINE_SMART_PTR_TYPE_ALIASES(CommandBuffer);
    _BV_DEFINE_SMART_PTR_TYPE_ALIASES(CommandPool);
    _BV_DEFINE_SMART_PTR_TYPE_ALIASES(Semaphore);
    _BV_DEFINE_SMART_PTR_TYPE_ALIASES(Fence);
    _BV_DEFINE_SMART_PTR_TYPE_ALIASES(Buffer);
    _BV_DEFINE_SMART_PTR_TYPE_ALIASES(DeviceMemory);
    _BV_DEFINE_SMART_PTR_TYPE_ALIASES(DescriptorSet);
    _BV_DEFINE_SMART_PTR_TYPE_ALIASES(DescriptorPool);
    _BV_DEFINE_SMART_PTR_TYPE_ALIASES(BufferView);
    _BV_DEFINE_SMART_PTR_TYPE_ALIASES(PipelineCache);
    _BV_DEFINE_SMART_PTR_TYPE_ALIASES(MemoryRegion);
    _BV_DEFINE_SMART_PTR_TYPE_ALIASES(MemoryChunk);
    _BV_DEFINE_SMART_PTR_TYPE_ALIASES(MemoryBank);

#pragma region data-only structs and enums

    template<typename Enum>
    using EnumStrMap = std::unordered_map<Enum, std::string>;

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_MAKE_API_VERSION.html
    struct Version
    {
        uint8_t variant : 8 = 0;
        uint8_t major : 8 = 0;
        uint8_t minor : 8 = 0;
        uint8_t patch : 8 = 0;

        Version() = default;

        constexpr Version(
            uint8_t variant,
            uint8_t major,
            uint8_t minor,
            uint8_t patch
        )
            : variant(variant), major(major), minor(minor), patch(patch)
        {}

        constexpr Version(uint32_t encoded)
            : variant(VK_API_VERSION_VARIANT(encoded)),
            major(VK_API_VERSION_MAJOR(encoded)),
            minor(VK_API_VERSION_MINOR(encoded)),
            patch(VK_API_VERSION_PATCH(encoded))
        {}

        std::string to_string() const;

        constexpr uint32_t encode() const
        {
            return VK_MAKE_API_VERSION(variant, major, minor, patch);
        }

    };

    static EnumStrMap<VkResult> VkResult_strmap{
        {
            VK_SUCCESS,
            "VK_SUCCESS: command successfully completed"
        },
{
    VK_NOT_READY,
    "VK_NOT_READY: a fence or query has not yet completed"
},
{
    VK_TIMEOUT,
    "VK_TIMEOUT: a wait operation has not completed in the specified time"
},
{
    VK_EVENT_SET,
    "VK_EVENT_SET: an event is signaled"
},
{
    VK_EVENT_RESET,
    "VK_EVENT_RESET: an event is unsignaled"
},
{
    VK_INCOMPLETE,
    "VK_INCOMPLETE: a return array was too small for the result"
},
{
    VK_ERROR_OUT_OF_HOST_MEMORY,
    "VK_ERROR_OUT_OF_HOST_MEMORY: a host memory allocation has failed"
},
{
    VK_ERROR_OUT_OF_DEVICE_MEMORY,
    "VK_ERROR_OUT_OF_DEVICE_MEMORY: a device memory allocation has failed"
},
{
    VK_ERROR_INITIALIZATION_FAILED,
    "VK_ERROR_INITIALIZATION_FAILED: initialization of an object could not be "
    "completed for implementation-specific reasons."
},
{
    VK_ERROR_DEVICE_LOST,
    "VK_ERROR_DEVICE_LOST: the logical or physical device has been lost"
},
{
    VK_ERROR_MEMORY_MAP_FAILED,
    "VK_ERROR_MEMORY_MAP_FAILED: mapping of a memory object has failed"
},
{
    VK_ERROR_LAYER_NOT_PRESENT,
    "VK_ERROR_LAYER_NOT_PRESENT: a requested layer is not present or could not "
    "be loaded."
},
{
    VK_ERROR_EXTENSION_NOT_PRESENT,
    "VK_ERROR_EXTENSION_NOT_PRESENT: a requested extension is not supported"
},
{
    VK_ERROR_FEATURE_NOT_PRESENT,
    "VK_ERROR_FEATURE_NOT_PRESENT: a requested feature is not supported"
},
{
    VK_ERROR_INCOMPATIBLE_DRIVER,
    "VK_ERROR_INCOMPATIBLE_DRIVER: the requested version of Vulkan is not "
    "supported by the driver or is otherwise incompatible for "
        "implementation-specific reasons."
},
{
    VK_ERROR_TOO_MANY_OBJECTS,
    "VK_ERROR_TOO_MANY_OBJECTS: too many objects of the type have already been "
    "created"
},
{
    VK_ERROR_FORMAT_NOT_SUPPORTED,
    "VK_ERROR_FORMAT_NOT_SUPPORTED: a requested format is not supported on "
    "this device."
},
{
    VK_ERROR_FRAGMENTED_POOL,
    "VK_ERROR_FRAGMENTED_POOL: a pool allocation has failed due to "
    "fragmentation of the pool's memory. this must only be returned if no "
        "attempt to allocate host or device memory was made to accommodate the "
        "new allocation. this should be returned in preference to "
        "VK_ERROR_OUT_OF_POOL_MEMORY, but only if the implementation is "
        "certain that the pool allocation failure was due to fragmentation."
},
{
    VK_ERROR_UNKNOWN,
    "VK_ERROR_UNKNOWN: an unknown error has occurred; either the application "
    "has provided invalid input, or an implementation failure has occurred."
},
{
    VK_ERROR_OUT_OF_POOL_MEMORY,
    "VK_ERROR_OUT_OF_POOL_MEMORY: a pool memory allocation has failed. this "
    "must only be returned if no attempt to allocate host or device memory was "
        "made to accommodate the new allocation. if the failure was definitely "
        "due to fragmentation of the pool, VK_ERROR_FRAGMENTED_POOL should be "
        "returned instead."
},
{
    VK_ERROR_INVALID_EXTERNAL_HANDLE,
    "VK_ERROR_INVALID_EXTERNAL_HANDLE: an external handle is not a valid "
    "handle of the specified type."
},
{
    VK_ERROR_FRAGMENTATION,
    "VK_ERROR_FRAGMENTATION: a descriptor pool creation has failed due to "
    "fragmentation"
},
{
    VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS,
    "VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS: a buffer creation or memory "
    "allocation failed because the requested address is not available. a "
        "shader group handle assignment failed because the requested shader "
        "group handle information is no longer valid."
},
{
    VK_PIPELINE_COMPILE_REQUIRED,
    "VK_PIPELINE_COMPILE_REQUIRED: a requested pipeline creation would have "
    "required compilation, but the application requested compilation to not be "
        "performed."
},
{
    VK_ERROR_SURFACE_LOST_KHR,
    "VK_ERROR_SURFACE_LOST_KHR: a surface is no longer available"
},
{
    VK_ERROR_NATIVE_WINDOW_IN_USE_KHR,
    "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR: the requested window is already in use "
    "by Vulkan or another API in a manner which prevents it from being used "
        "again."
},
{
    VK_SUBOPTIMAL_KHR,
    "VK_SUBOPTIMAL_KHR: a swapchain no longer matches the surface properties "
    "exactly, but can still be used to present to the surface successfully."
},
{
    VK_ERROR_OUT_OF_DATE_KHR,
    "VK_ERROR_OUT_OF_DATE_KHR: a surface has changed in such a way that it is "
    "no longer compatible with the swapchain, and further presentation "
        "requests using the swapchain will fail. applications must query the "
        "new surface properties and recreate their swapchain if they wish to "
        "continue presenting to the surface."
},
{
    VK_ERROR_INCOMPATIBLE_DISPLAY_KHR,
    "VK_ERROR_INCOMPATIBLE_DISPLAY_KHR: the display used by a swapchain does "
    "not use the same presentable image layout, or is incompatible in a way "
        "that prevents sharing an image."
},
{
    VK_ERROR_VALIDATION_FAILED_EXT,
    "VK_ERROR_VALIDATION_FAILED_EXT: a command failed because invalid usage "
    "was detected by the implementation or a validation-layer."
},
{
    VK_ERROR_INVALID_SHADER_NV,
    "VK_ERROR_INVALID_SHADER_NV: one or more shaders failed to compile or "
    "link. more details are reported back to the application via "
        "VK_EXT_debug_report if enabled."
},
{
    VK_ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR,
    "VK_ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR: the requested VkImageUsageFlags "
    "are not supported"
},
{
    VK_ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR,
    "VK_ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR: the requested video "
    "picture layout is not supported."
},
{
    VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR,
    "VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR: a video profile "
    "operation specified via VkVideoProfileInfoKHR::videoCodecOperation is not "
        "supported."
},
{
    VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR,
    "VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR: format parameters in a "
    "requested VkVideoProfileInfoKHR chain are not supported."
},
{
    VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR,
    "VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR: codec-specific parameters "
    "in a requested VkVideoProfileInfoKHR chain are not supported."
},
{
    VK_ERROR_VIDEO_STD_VERSION_NOT_SUPPORTED_KHR,
    "VK_ERROR_VIDEO_STD_VERSION_NOT_SUPPORTED_KHR: the specified video Std "
    "header version is not supported."
},
{
    VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT,
    "VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT"
},
{
    VK_ERROR_NOT_PERMITTED_KHR,
    "VK_ERROR_NOT_PERMITTED_KHR: the driver implementation has denied a "
    "request to acquire a priority above the default priority "
        "(VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_EXT) because the application does "
        "not have sufficient privileges."
},
{
    VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT,
    "VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT: an operation on a swapchain "
    "created with VK_FULL_SCREEN_EXCLUSIVE_APPLICATION_CONTROLLED_EXT failed "
        "as it did not have exclusive full-screen access. this may occur due "
        "to implementation-dependent reasons, outside of the application's "
        "control."
},
{
    VK_THREAD_IDLE_KHR,
    "VK_THREAD_IDLE_KHR: a deferred operation is not complete but there is "
    "currently no work for this thread to do at the time of this call."
},
{
    VK_THREAD_DONE_KHR,
    "VK_THREAD_DONE_KHR: a deferred operation is not complete but there is no "
    "work remaining to assign to additional threads."
},
{
    VK_OPERATION_DEFERRED_KHR,
    "VK_OPERATION_DEFERRED_KHR: a deferred operation was requested and at "
    "least some of the work was deferred."
},
{
    VK_OPERATION_NOT_DEFERRED_KHR,
    "VK_OPERATION_NOT_DEFERRED_KHR: a deferred operation was requested and no "
    "operations were deferred."
},
{
    VK_ERROR_COMPRESSION_EXHAUSTED_EXT,
    "VK_ERROR_COMPRESSION_EXHAUSTED_EXT: an image creation failed because "
    "internal resources required for compression are exhausted. this must only "
        "be returned when fixed-rate compression is requested."
},
{
    VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT,
    "VK_INCOMPATIBLE_SHADER_BINARY_EXT : the provided binary shader code "
    "is not compatible with this device."
}
    };

    std::string VkResult_to_string(VkResult result);

    enum class VulkanApiVersion
    {
        Vulkan1_0,
        Vulkan1_1,
        Vulkan1_2,
        Vulkan1_3
    };

    uint32_t VulkanApiVersion_encode(VulkanApiVersion version);

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkExtensionProperties.html
    struct ExtensionProperties
    {
        std::string name;
        uint32_t spec_version;
    };

    ExtensionProperties ExtensionProperties_from_vk(
        VkExtensionProperties vk_properties
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkLayerProperties.html
    struct LayerProperties
    {
        std::string name;
        Version spec_version;
        uint32_t implementation_version;
        std::string description;
    };

    LayerProperties LayerProperties_from_vk(
        const VkLayerProperties& vk_properties
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceLimits.html
    struct PhysicalDeviceLimits
    {
        uint32_t max_image_dimension_1d;
        uint32_t max_image_dimension_2d;
        uint32_t max_image_dimension_3d;
        uint32_t max_image_dimension_cube;
        uint32_t max_image_array_layers;
        uint32_t max_texel_buffer_elements;
        uint32_t max_uniform_buffer_range;
        uint32_t max_storage_buffer_range;
        uint32_t max_push_constants_size;
        uint32_t max_memory_allocation_count;
        uint32_t max_sampler_allocation_count;
        uint64_t buffer_image_granularity;
        uint64_t sparse_address_space_size;
        uint32_t max_bound_descriptor_sets;
        uint32_t max_per_stage_descriptor_samplers;
        uint32_t max_per_stage_descriptor_uniform_buffers;
        uint32_t max_per_stage_descriptor_storage_buffers;
        uint32_t max_per_stage_descriptor_sampled_images;
        uint32_t max_per_stage_descriptor_storage_images;
        uint32_t max_per_stage_descriptor_input_attachments;
        uint32_t max_per_stage_resources;
        uint32_t max_descriptor_set_samplers;
        uint32_t max_descriptor_set_uniform_buffers;
        uint32_t max_descriptor_set_uniform_buffers_dynamic;
        uint32_t max_descriptor_set_storage_buffers;
        uint32_t max_descriptor_set_storage_buffers_dynamic;
        uint32_t max_descriptor_set_sampled_images;
        uint32_t max_descriptor_set_storage_images;
        uint32_t max_descriptor_set_input_attachments;
        uint32_t max_vertex_input_attributes;
        uint32_t max_vertex_input_bindings;
        uint32_t max_vertex_input_attribute_offset;
        uint32_t max_vertex_input_binding_stride;
        uint32_t max_vertex_output_components;
        uint32_t max_tessellation_generation_level;
        uint32_t max_tessellation_patch_size;
        uint32_t max_tessellation_control_per_vertex_input_components;
        uint32_t max_tessellation_control_per_vertex_output_components;
        uint32_t max_tessellation_control_per_patch_output_components;
        uint32_t max_tessellation_control_total_output_components;
        uint32_t max_tessellation_evaluation_input_components;
        uint32_t max_tessellation_evaluation_output_components;
        uint32_t max_geometry_shader_invocations;
        uint32_t max_geometry_input_components;
        uint32_t max_geometry_output_components;
        uint32_t max_geometry_output_vertices;
        uint32_t max_geometry_total_output_components;
        uint32_t max_fragment_input_components;
        uint32_t max_fragment_output_attachments;
        uint32_t max_fragment_dual_src_attachments;
        uint32_t max_fragment_combined_output_resources;
        uint32_t max_compute_shared_memory_size;
        std::array<uint32_t, 3> max_compute_work_group_count;
        uint32_t max_compute_work_group_invocations;
        std::array<uint32_t, 3> max_compute_work_group_size;
        uint32_t sub_pixel_precision_bits;
        uint32_t sub_texel_precision_bits;
        uint32_t mipmap_precision_bits;
        uint32_t max_draw_indexed_index_value;
        uint32_t max_draw_indirect_count;
        float max_sampler_lod_bias;
        float max_sampler_anisotropy;
        uint32_t max_viewports;
        std::array<uint32_t, 2> max_viewport_dimensions;
        std::array<float, 2> viewport_bounds_range;
        uint32_t viewport_sub_pixel_bits;
        size_t min_memory_map_alignment;
        uint64_t min_texel_buffer_offset_alignment;
        uint64_t min_uniform_buffer_offset_alignment;
        uint64_t min_storage_buffer_offset_alignment;
        int32_t min_texel_offset;
        uint32_t max_texel_offset;
        int32_t min_texel_gather_offset;
        uint32_t max_texel_gather_offset;
        float min_interpolation_offset;
        float max_interpolation_offset;
        uint32_t sub_pixel_interpolation_offset_bits;
        uint32_t max_framebuffer_width;
        uint32_t max_framebuffer_height;
        uint32_t max_framebuffer_layers;
        VkSampleCountFlags framebuffer_color_sample_counts;
        VkSampleCountFlags framebuffer_depth_sample_counts;
        VkSampleCountFlags framebuffer_stencil_sample_counts;
        VkSampleCountFlags framebuffer_no_attachments_sample_counts;
        uint32_t max_color_attachments;
        VkSampleCountFlags sampled_image_color_sample_counts;
        VkSampleCountFlags sampled_image_integer_sample_counts;
        VkSampleCountFlags sampled_image_depth_sample_counts;
        VkSampleCountFlags sampled_image_stencil_sample_counts;
        VkSampleCountFlags storage_image_sample_counts;
        uint32_t max_sample_mask_words;
        bool timestamp_compute_and_graphics;
        float timestamp_period;
        uint32_t max_clip_distances;
        uint32_t max_cull_distances;
        uint32_t max_combined_clip_and_cull_distances;
        uint32_t discrete_queue_priorities;
        std::array<float, 2> point_size_range;
        std::array<float, 2> line_width_range;
        float point_size_granularity;
        float line_width_granularity;
        bool strict_lines;
        bool standard_sample_locations;
        uint64_t optimal_buffer_copy_offset_alignment;
        uint64_t optimal_buffer_copy_row_pitch_alignment;
        uint64_t non_coherent_atom_size;
    };

    PhysicalDeviceLimits PhysicalDeviceLimits_from_vk(
        const VkPhysicalDeviceLimits& vk_limits
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceSparseProperties.html
    struct PhysicalDeviceSparseProperties
    {
        bool residency_standard_2d_block_shape : 1;
        bool residency_standard_2d_multisample_block_shape : 1;
        bool residency_standard_3d_block_shape : 1;
        bool residency_aligned_mip_size : 1;
        bool residency_non_resident_strict : 1;
    };

    PhysicalDeviceSparseProperties PhysicalDeviceSparseProperties_from_vk(
        const VkPhysicalDeviceSparseProperties& vk_properties
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceProperties.html
    struct PhysicalDeviceProperties
    {
        Version api_version;
        uint32_t driver_version;
        uint32_t vendor_id;
        uint32_t device_id;
        VkPhysicalDeviceType device_type;
        std::string device_name;
        std::array<uint8_t, VK_UUID_SIZE> pipeline_cache_uuid;
        PhysicalDeviceLimits limits;
        PhysicalDeviceSparseProperties sparse_properties;
    };

    PhysicalDeviceProperties PhysicalDeviceProperties_from_vk(
        const VkPhysicalDeviceProperties& vk_properties
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceFeatures.html
    struct PhysicalDeviceFeatures
    {
        bool robust_buffer_access : 1 = false;
        bool full_draw_index_uint32 : 1 = false;
        bool image_cube_array : 1 = false;
        bool independent_blend : 1 = false;
        bool geometry_shader : 1 = false;
        bool tessellation_shader : 1 = false;
        bool sample_rate_shading : 1 = false;
        bool dual_src_blend : 1 = false;
        bool logic_op : 1 = false;
        bool multi_draw_indirect : 1 = false;
        bool draw_indirect_first_instance : 1 = false;
        bool depth_clamp : 1 = false;
        bool depth_bias_clamp : 1 = false;
        bool fill_mode_non_solid : 1 = false;
        bool depth_bounds : 1 = false;
        bool wide_lines : 1 = false;
        bool large_points : 1 = false;
        bool alpha_to_one : 1 = false;
        bool multi_viewport : 1 = false;
        bool sampler_anisotropy : 1 = false;
        bool texture_compression_etc2 : 1 = false;
        bool texture_compression_astc_ldr : 1 = false;
        bool texture_compression_bc : 1 = false;
        bool occlusion_query_precise : 1 = false;
        bool pipeline_statistics_query : 1 = false;
        bool vertex_pipeline_stores_and_atomics : 1 = false;
        bool fragment_stores_and_atomics : 1 = false;
        bool shader_tessellation_and_geometry_point_size : 1 = false;
        bool shader_image_gather_extended : 1 = false;
        bool shader_storage_image_extended_formats : 1 = false;
        bool shader_storage_image_multisample : 1 = false;
        bool shader_storage_image_read_without_format : 1 = false;
        bool shader_storage_image_write_without_format : 1 = false;
        bool shader_uniform_buffer_array_dynamic_indexing : 1 = false;
        bool shader_sampled_image_array_dynamic_indexing : 1 = false;
        bool shader_storage_buffer_array_dynamic_indexing : 1 = false;
        bool shader_storage_image_array_dynamic_indexing : 1 = false;
        bool shader_clip_distance : 1 = false;
        bool shader_cull_distance : 1 = false;
        bool shader_float64 : 1 = false;
        bool shader_int64 : 1 = false;
        bool shader_int16 : 1 = false;
        bool shader_resource_residency : 1 = false;
        bool shader_resource_min_lod : 1 = false;
        bool sparse_binding : 1 = false;
        bool sparse_residency_buffer : 1 = false;
        bool sparse_residency_image_2d : 1 = false;
        bool sparse_residency_image_3d : 1 = false;
        bool sparse_residency2_samples : 1 = false;
        bool sparse_residency4_samples : 1 = false;
        bool sparse_residency8_samples : 1 = false;
        bool sparse_residency16_samples : 1 = false;
        bool sparse_residency_aliased : 1 = false;
        bool variable_multisample_rate : 1 = false;
        bool inherited_queries : 1 = false;
    };

    PhysicalDeviceFeatures PhysicalDeviceFeatures_from_vk(
        const VkPhysicalDeviceFeatures& vk_features
    );

    VkPhysicalDeviceFeatures PhysicalDeviceFeatures_to_vk(
        const PhysicalDeviceFeatures& features
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkExtent3D.html
    struct Extent3d
    {
        uint32_t width;
        uint32_t height;
        uint32_t depth;
    };

    Extent3d Extent3d_from_vk(const VkExtent3D& vk_extent_3d);
    VkExtent3D Extent3d_to_vk(const Extent3d& extent_3d);

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkExtent2D.html
    struct Extent2d
    {
        uint32_t width;
        uint32_t height;
    };

    Extent2d Extent2d_from_vk(const VkExtent2D& vk_extent_2d);
    VkExtent2D Extent2d_to_vk(const Extent2d& extent_2d);

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkQueueFamilyProperties.html
    struct QueueFamily
    {
        VkQueueFlags queue_flags;
        uint32_t queue_count;
        uint32_t timestamp_valid_bits;
        Extent3d min_image_transfer_granularity;
    };

    QueueFamily QueueFamily_from_vk(const VkQueueFamilyProperties& vk_family);

    // provided by VK_KHR_surface
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSurfaceCapabilitiesKHR.html
    struct SurfaceCapabilities
    {
        uint32_t min_image_count;
        uint32_t max_image_count;
        Extent2d current_extent;
        Extent2d min_image_extent;
        Extent2d max_image_extent;
        uint32_t max_image_array_layers;
        VkSurfaceTransformFlagsKHR supported_transforms;
        VkSurfaceTransformFlagBitsKHR current_transform;
        VkCompositeAlphaFlagsKHR supported_composite_alpha;
        VkImageUsageFlags supported_usage_flags;
    };

    SurfaceCapabilities SurfaceCapabilities_from_vk(
        const VkSurfaceCapabilitiesKHR& vk_capabilities
    );

    // provided by VK_KHR_surface
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSurfaceFormatKHR.html
    struct SurfaceFormat
    {
        VkFormat format;
        VkColorSpaceKHR color_space;
    };

    SurfaceFormat SurfaceFormat_from_vk(
        const VkSurfaceFormatKHR& vk_surface_format
    );

    // provided by VK_KHR_surface
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetPhysicalDeviceSurfaceCapabilitiesKHR.html
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetPhysicalDeviceSurfaceFormatsKHR.html
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetPhysicalDeviceSurfacePresentModesKHR.html
    struct SwapchainSupport
    {
        SurfaceCapabilities capabilities;
        std::vector<SurfaceFormat> surface_formats;
        std::vector<VkPresentModeKHR> present_modes;
    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkInstanceCreateInfo.html
    struct ContextConfig
    {
        // enables the VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR flag
        // which specifies that the instance will enumerate available Vulkan
        // Portability-compliant physical devices and groups in addition to the
        // Vulkan physical devices and groups that are enumerated by default.
        // you might want to enable the VK_KHR_portability_enumeration extension
        // when using this.
        bool will_enumerate_portability = false;

        std::string app_name;
        Version app_version;

        std::string engine_name;
        Version engine_version;

        VulkanApiVersion vulkan_api_version;

        std::vector<std::string> layers;
        std::vector<std::string> extensions;
    };

    // provided by VK_EXT_debug_utils
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDebugUtilsLabelEXT.html
    struct DebugLabel
    {
        std::string name;
        std::array<float, 4> color;
    };

    DebugLabel DebugLabel_from_vk(const VkDebugUtilsLabelEXT& vk_label);

    // provided by VK_EXT_debug_utils
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDebugUtilsObjectNameInfoEXT.html
    struct DebugObjectInfo
    {
        VkObjectType type;
        uint64_t handle;
        std::string name;
    };

    DebugObjectInfo DebugObjectInfo_from_vk(
        const VkDebugUtilsObjectNameInfoEXT& vk_info
    );

    // provided by VK_EXT_debug_utils
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDebugUtilsMessengerCallbackDataEXT.html
    struct DebugMessageData
    {
        std::string message_id_name;
        int32_t message_id_number;
        std::string message;
        std::vector<DebugLabel> queue_labels;
        std::vector<DebugLabel> cmd_buf_labels;
        std::vector<DebugObjectInfo> objects;
    };

    DebugMessageData DebugMessageData_from_vk(
        const VkDebugUtilsMessengerCallbackDataEXT& vk_data
    );

    // provided by VK_EXT_debug_utils
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/PFN_vkDebugUtilsMessengerCallbackEXT.html
    using DebugCallback = std::function<void(
        VkDebugUtilsMessageSeverityFlagBitsEXT,
        VkDebugUtilsMessageTypeFlagsEXT,
        const DebugMessageData&
        )>;

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDeviceQueueCreateInfo.html
    struct QueueRequest
    {
        VkDeviceQueueCreateFlags flags;
        uint32_t queue_family_index;
        uint32_t num_queues_to_create;
        std::vector<float> priorities; // * same size as num_queues_to_create
    };

    VkDeviceQueueCreateInfo QueueRequest_to_vk(
        const QueueRequest& request,
        std::vector<float>& waste_priorities
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDeviceCreateInfo.html
    struct DeviceConfig
    {
        std::vector<QueueRequest> queue_requests;
        std::vector<std::string> extensions;
        PhysicalDeviceFeatures enabled_features;
    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageCreateInfo.html
    struct ImageConfig
    {
        VkImageCreateFlags flags;
        VkImageType image_type;
        VkFormat format;
        Extent3d extent;
        uint32_t mip_levels;
        uint32_t array_layers;
        VkSampleCountFlagBits samples;
        VkImageTiling tiling;
        VkImageUsageFlags usage;
        VkSharingMode sharing_mode;
        std::vector<uint32_t> queue_family_indices;
        VkImageLayout initial_layout;
    };

    // provided by VK_KHR_swapchain
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSwapchainCreateInfoKHR.html
    struct SwapchainConfig
    {
        VkSwapchainCreateFlagsKHR flags;
        uint32_t min_image_count;
        VkFormat image_format;
        VkColorSpaceKHR image_color_space;
        Extent2d image_extent;
        uint32_t image_array_layers;
        VkImageUsageFlags image_usage;
        VkSharingMode image_sharing_mode;
        std::vector<uint32_t> queue_family_indices;
        VkSurfaceTransformFlagBitsKHR pre_transform;
        VkCompositeAlphaFlagBitsKHR composite_alpha;
        VkPresentModeKHR present_mode;
        bool clipped;
    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkComponentMapping.html
    struct ComponentMapping
    {
        VkComponentSwizzle r = VK_COMPONENT_SWIZZLE_IDENTITY;
        VkComponentSwizzle g = VK_COMPONENT_SWIZZLE_IDENTITY;
        VkComponentSwizzle b = VK_COMPONENT_SWIZZLE_IDENTITY;
        VkComponentSwizzle a = VK_COMPONENT_SWIZZLE_IDENTITY;
    };

    VkComponentMapping ComponentMapping_to_vk(const ComponentMapping& mapping);

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageSubresourceRange.html
    struct ImageSubresourceRange
    {
        VkImageAspectFlags aspect_mask;
        uint32_t base_mip_level;
        uint32_t level_count;
        uint32_t base_array_layer;
        uint32_t layer_count;
    };

    VkImageSubresourceRange ImageSubresourceRange_to_vk(
        const ImageSubresourceRange& range
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageViewCreateInfo.html
    struct ImageViewConfig
    {
        VkImageViewCreateFlags flags;
        VkImageViewType view_type;
        VkFormat format;
        ComponentMapping components;
        ImageSubresourceRange subresource_range;
    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSpecializationMapEntry.html
    struct SpecializationMapEntry
    {
        uint32_t constant_id;
        uint32_t offset;
        size_t size;
    };

    VkSpecializationMapEntry SpecializationMapEntry_to_vk(
        const SpecializationMapEntry& entry
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSpecializationInfo.html
    struct SpecializationInfo
    {
        std::vector<SpecializationMapEntry> map_entries;

        // std::copy your data here (and use reinterpret_cast)
        std::vector<uint8_t> data;
    };

    VkSpecializationInfo SpecializationInfo_to_vk(
        const SpecializationInfo& info,
        std::vector<VkSpecializationMapEntry>& waste_vk_map_entries,
        std::vector<uint8_t>& waste_data
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineShaderStageCreateInfo.html
    struct ShaderStage
    {
        VkPipelineShaderStageCreateFlags flags;
        VkShaderStageFlagBits stage;
        ShaderModuleWPtr module;
        std::string entry_point;
        std::optional<SpecializationInfo> specialization_info;
    };

    VkPipelineShaderStageCreateInfo ShaderStage_to_vk(
        const ShaderStage& stage,
        VkSpecializationInfo& waste_vk_specialization_info,
        std::vector<VkSpecializationMapEntry>& waste_vk_map_entries,
        std::vector<uint8_t>& waste_data
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineDynamicStateCreateInfo.html
    using DynamicStates = std::vector<VkDynamicState>;

    VkPipelineDynamicStateCreateInfo DynamicStates_to_vk(
        const DynamicStates& states,
        std::vector<VkDynamicState>& waste_dynamic_states
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkVertexInputBindingDescription.html
    struct VertexInputBindingDescription
    {
        uint32_t binding;
        uint32_t stride;
        VkVertexInputRate input_rate;
    };

    VkVertexInputBindingDescription VertexInputBindingDescription_to_vk(
        const VertexInputBindingDescription& description
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkVertexInputAttributeDescription.html
    struct VertexInputAttributeDescription
    {
        uint32_t location;
        uint32_t binding;
        VkFormat format;
        uint32_t offset;
    };

    VkVertexInputAttributeDescription VertexInputAttributeDescription_to_vk(
        const VertexInputAttributeDescription& description
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineVertexInputStateCreateInfo.html
    struct VertexInputState
    {
        std::vector<VertexInputBindingDescription> binding_descriptions;
        std::vector<VertexInputAttributeDescription> attribute_descriptions;
    };

    VkPipelineVertexInputStateCreateInfo VertexInputState_to_vk(
        const VertexInputState& state,

        std::vector<VkVertexInputBindingDescription>&
        waste_vk_binding_descriptions,

        std::vector<VkVertexInputAttributeDescription>&
        waste_vk_attribute_descriptions
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineInputAssemblyStateCreateInfo.html
    struct InputAssemblyState
    {
        VkPrimitiveTopology topology;
        bool primitive_restart_enable;
    };

    VkPipelineInputAssemblyStateCreateInfo InputAssemblyState_to_vk(
        const InputAssemblyState& state
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineTessellationStateCreateInfo.html
    struct TessellationState
    {
        uint32_t patch_control_points;
    };

    VkPipelineTessellationStateCreateInfo TessellationState_to_vk(
        const TessellationState& state
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkViewport.html
    struct Viewport
    {
        float x;
        float y;
        float width;
        float height;
        float min_depth;
        float max_depth;
    };

    VkViewport Viewport_to_vk(const Viewport& viewport);

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkOffset2D.html
    struct Offset2d
    {
        int32_t x;
        int32_t y;
    };

    VkOffset2D Offset2d_to_vk(const Offset2d& offset);

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkOffset3D.html
    struct Offset3d
    {
        int32_t x;
        int32_t y;
        int32_t z;
    };

    VkOffset3D Offset3d_to_vk(const Offset3d& offset);

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkRect2D.html
    struct Rect2d
    {
        Offset2d offset;
        Extent2d extent;
    };

    VkRect2D Rect2d_to_vk(const Rect2d& rect);

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineViewportStateCreateInfo.html
    struct ViewportState
    {
        std::vector<Viewport> viewports;
        std::vector<Rect2d> scissors;
    };

    VkPipelineViewportStateCreateInfo ViewportState_to_vk(
        const ViewportState& state,
        std::vector<VkViewport>& waste_vk_viewports,
        std::vector<VkRect2D>& waste_vk_scissors
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineRasterizationStateCreateInfo.html
    struct RasterizationState
    {
        bool depth_clamp_enable;
        bool rasterizer_discard_enable;
        VkPolygonMode polygon_mode;
        VkCullModeFlags cull_mode;
        VkFrontFace front_face;
        bool depth_bias_enable;
        float depth_bias_constant_factor;
        float depth_bias_clamp;
        float depth_bias_slope_factor;
        float line_width;
    };

    VkPipelineRasterizationStateCreateInfo RasterizationState_to_vk(
        const RasterizationState& state
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineMultisampleStateCreateInfo.html
    struct MultisampleState
    {
        VkSampleCountFlagBits rasterization_samples;
        bool sample_shading_enable;
        float min_sample_shading;
        std::vector<VkSampleMask> sample_mask;
        bool alpha_to_coverage_enable;
        bool alpha_to_one_enable;
    };

    VkPipelineMultisampleStateCreateInfo MultisampleState_to_vk(
        const MultisampleState& state,
        std::vector<VkSampleMask>& waste_sample_mask
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkStencilOpState.html
    struct StencilOpState
    {
        VkStencilOp fail_op;
        VkStencilOp pass_op;
        VkStencilOp depth_fail_op;
        VkCompareOp compare_op;
        uint32_t compare_mask;
        uint32_t write_mask;
        uint32_t reference;
    };

    VkStencilOpState StencilOpState_to_vk(const StencilOpState& state);

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineDepthStencilStateCreateInfo.html
    struct DepthStencilState
    {
        VkPipelineDepthStencilStateCreateFlags flags;
        bool depth_test_enable;
        bool depth_write_enable;
        VkCompareOp depth_compare_op;
        bool depth_bounds_test_enable;
        bool stencil_test_enable;
        StencilOpState front;
        StencilOpState back;
        float min_depth_bounds;
        float max_depth_bounds;
    };

    VkPipelineDepthStencilStateCreateInfo DepthStencilState_to_vk(
        const DepthStencilState& state
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineColorBlendAttachmentState.html
    struct ColorBlendAttachment
    {
        bool blend_enable;
        VkBlendFactor src_color_blend_factor;
        VkBlendFactor dst_color_blend_factor;
        VkBlendOp color_blend_op;
        VkBlendFactor src_alpha_blend_factor;
        VkBlendFactor dst_alpha_blend_factor;
        VkBlendOp alpha_blend_op;
        VkColorComponentFlags color_write_mask;
    };

    VkPipelineColorBlendAttachmentState ColorBlendAttachment_to_vk(
        const ColorBlendAttachment& attachment
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineColorBlendStateCreateInfo.html
    struct ColorBlendState
    {
        VkPipelineColorBlendStateCreateFlags flags;
        bool logic_op_enable;
        VkLogicOp logic_op;
        std::vector<ColorBlendAttachment> attachments;
        std::array<float, 4> blend_constants;
    };

    VkPipelineColorBlendStateCreateInfo ColorBlendState_to_vk(
        const ColorBlendState& state,

        std::vector<VkPipelineColorBlendAttachmentState>&
        waste_vk_color_blend_attachments
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSamplerCreateInfo.html
    struct SamplerConfig
    {
        VkSamplerCreateFlags flags;
        VkFilter mag_filter;
        VkFilter min_filter;
        VkSamplerMipmapMode mipmap_mode;
        VkSamplerAddressMode address_mode_u;
        VkSamplerAddressMode address_mode_v;
        VkSamplerAddressMode address_mode_w;
        float mip_lod_bias;
        bool anisotropy_enable;
        float max_anisotropy;
        bool compare_enable;
        VkCompareOp compare_op;
        float min_lod;
        float max_lod;
        VkBorderColor border_color;
        bool unnormalized_coordinates;
    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDescriptorSetLayoutBinding.html
    struct DescriptorSetLayoutBinding
    {
        uint32_t binding;
        VkDescriptorType descriptor_type;
        uint32_t descriptor_count;
        VkShaderStageFlags stage_flags;
        std::vector<SamplerWPtr> immutable_samplers;
    };

    VkDescriptorSetLayoutBinding DescriptorSetLayoutBinding_to_vk(
        const DescriptorSetLayoutBinding& binding,
        std::vector<VkSampler>& waste_vk_immutable_samplers
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDescriptorSetLayoutCreateInfo.html
    struct DescriptorSetLayoutConfig
    {
        VkDescriptorSetLayoutCreateFlags flags;
        std::vector<DescriptorSetLayoutBinding> bindings;
    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPushConstantRange.html
    struct PushConstantRange
    {
        VkShaderStageFlags stage_flags;
        uint32_t offset;
        uint32_t size;
    };

    VkPushConstantRange PushConstantRange_to_vk(const PushConstantRange& range);

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineLayoutCreateInfo.html
    struct PipelineLayoutConfig
    {
        VkPipelineLayoutCreateFlags flags;
        std::vector<DescriptorSetLayoutWPtr> set_layouts;
        std::vector<PushConstantRange> push_constant_ranges;
    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkAttachmentDescription.html
    struct Attachment
    {
        VkAttachmentDescriptionFlags flags;
        VkFormat format;
        VkSampleCountFlagBits samples;
        VkAttachmentLoadOp load_op;
        VkAttachmentStoreOp store_op;
        VkAttachmentLoadOp stencil_load_op;
        VkAttachmentStoreOp stencil_store_op;
        VkImageLayout initial_layout;
        VkImageLayout final_layout;
    };

    VkAttachmentDescription Attachment_to_vk(
        const Attachment& attachment
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkAttachmentReference.html
    struct AttachmentReference
    {
        uint32_t attachment;
        VkImageLayout layout;
    };

    VkAttachmentReference AttachmentReference_to_vk(
        const AttachmentReference& ref
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDescription.html
    struct Subpass
    {
        VkSubpassDescriptionFlags flags;
        VkPipelineBindPoint pipeline_bind_point;
        std::vector<AttachmentReference> input_attachments;
        std::vector<AttachmentReference> color_attachments;
        std::vector<AttachmentReference> resolve_attachments;
        std::optional<AttachmentReference> depth_stencil_attachment;
        std::vector<uint32_t> preserve_attachment_indices;
    };

    VkSubpassDescription Subpass_to_vk(
        const Subpass& subpass,
        std::vector<VkAttachmentReference>& waste_vk_input_attachments,
        std::vector<VkAttachmentReference>& waste_vk_color_attachments,
        std::vector<VkAttachmentReference>& waste_vk_resolve_attachments,
        VkAttachmentReference& waste_vk_depth_stencil_attachment,
        std::vector<uint32_t>& waste_preserve_attachment_indices
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubpassDependency.html
    struct SubpassDependency
    {
        uint32_t src_subpass;
        uint32_t dst_subpass;
        VkPipelineStageFlags src_stage_mask;
        VkPipelineStageFlags dst_stage_mask;
        VkAccessFlags src_access_mask;
        VkAccessFlags dst_access_mask;
        VkDependencyFlags dependency_flags;
    };

    VkSubpassDependency SubpassDependency_to_vk(const SubpassDependency& dep);

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkRenderPassCreateInfo.html
    struct RenderPassConfig
    {
        VkRenderPassCreateFlags flags;
        std::vector<Attachment> attachments;
        std::vector<Subpass> subpasses;
        std::vector<SubpassDependency> dependencies;
    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkGraphicsPipelineCreateInfo.html
    struct GraphicsPipelineConfig
    {
        VkPipelineCreateFlags flags;
        std::vector<ShaderStage> stages;
        std::optional<VertexInputState> vertex_input_state;
        std::optional<InputAssemblyState> input_assembly_state;
        std::optional<TessellationState> tessellation_state;
        std::optional<ViewportState> viewport_state;
        std::optional<RasterizationState> rasterization_state;
        std::optional<MultisampleState> multisample_state;
        std::optional<DepthStencilState> depth_stencil_state;
        std::optional<ColorBlendState> color_blend_state;
        DynamicStates dynamic_states;
        PipelineLayoutWPtr layout;
        RenderPassWPtr render_pass;
        uint32_t subpass_index;
        std::optional<GraphicsPipelineWPtr> base_pipeline;
    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkComputePipelineCreateInfo.html
    struct ComputePipelineConfig
    {
        VkPipelineCreateFlags flags;
        ShaderStage stage;
        PipelineLayoutWPtr layout;
        std::optional<GraphicsPipelineWPtr> base_pipeline;
    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html
    struct FramebufferConfig
    {
        VkFramebufferCreateFlags flags;
        RenderPassWPtr render_pass;
        std::vector<ImageViewWPtr> attachments;
        uint32_t width;
        uint32_t height;
        uint32_t layers;
    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkCommandPoolCreateInfo.html
    struct CommandPoolConfig
    {
        VkCommandPoolCreateFlags flags;
        uint32_t queue_family_index;
    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkCommandBufferInheritanceInfo.html
    struct CommandBufferInheritance
    {
        RenderPassWPtr render_pass;
        uint32_t subpass_index;
        std::optional<FramebufferWPtr> framebuffer;
        bool occlusion_query_enable;
        VkQueryControlFlags query_flags;
        VkQueryPipelineStatisticFlags pipeline_statistics;
    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkBufferCreateInfo.html
    struct BufferConfig
    {
        VkBufferCreateFlags flags;
        VkDeviceSize size;
        VkBufferUsageFlags usage;
        VkSharingMode sharing_mode;
        std::vector<uint32_t> queue_family_indices;
    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkMemoryRequirements.html
    struct MemoryRequirements
    {
        VkDeviceSize size;
        VkDeviceSize alignment;
        uint32_t memory_type_bits;
    };

    MemoryRequirements MemoryRequirements_from_vk(
        const VkMemoryRequirements& req
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkMemoryType.html
    struct MemoryType
    {
        VkMemoryPropertyFlags property_flags;
        uint32_t heap_index;
    };

    MemoryType MemoryType_from_vk(const VkMemoryType& type);

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkMemoryHeap.html
    struct MemoryHeap
    {
        VkDeviceSize size;
        VkMemoryHeapFlags flags;
    };

    MemoryHeap MemoryHeap_from_vk(const VkMemoryHeap& heap);

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceMemoryProperties.html
    struct PhysicalDeviceMemoryProperties
    {
        std::vector<MemoryType> memory_types;
        std::vector<MemoryHeap> memory_heaps;
    };

    PhysicalDeviceMemoryProperties PhysicalDeviceMemoryProperties_from_vk(
        const VkPhysicalDeviceMemoryProperties& properties
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkMemoryAllocateInfo.html
    struct DeviceMemoryConfig
    {
        VkDeviceSize allocation_size;
        uint32_t memory_type_index;
    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDescriptorPoolSize.html
    struct DescriptorPoolSize
    {
        VkDescriptorType type;
        uint32_t descriptor_count;
    };

    VkDescriptorPoolSize DescriptorPoolSize_to_vk(
        const DescriptorPoolSize& pool_size
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDescriptorPoolCreateInfo.html
    struct DescriptorPoolConfig
    {
        VkDescriptorPoolCreateFlags flags;
        uint32_t max_sets;
        std::vector<DescriptorPoolSize> pool_sizes;
    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDescriptorImageInfo.html
    struct DescriptorImageInfo
    {
        std::optional<SamplerWPtr> sampler;
        std::optional<ImageViewWPtr> image_view;
        VkImageLayout image_layout;
    };

    VkDescriptorImageInfo DescriptorImageInfo_to_vk(
        const DescriptorImageInfo& info
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDescriptorBufferInfo.html
    struct DescriptorBufferInfo
    {
        BufferWPtr buffer;
        VkDeviceSize offset;
        VkDeviceSize range;
    };

    VkDescriptorBufferInfo DescriptorBufferInfo_to_vk(
        const DescriptorBufferInfo& info
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkWriteDescriptorSet.html
    struct WriteDescriptorSet
    {
        DescriptorSetWPtr dst_set;
        uint32_t dst_binding;
        uint32_t dst_array_element;
        uint32_t descriptor_count;
        VkDescriptorType descriptor_type;
        std::vector<DescriptorImageInfo> image_infos;
        std::vector<DescriptorBufferInfo> buffer_infos;
        std::vector<BufferViewWPtr> texel_buffer_views;
    };

    VkWriteDescriptorSet WriteDescriptorSet_to_vk(
        const WriteDescriptorSet& write,
        std::vector<VkDescriptorImageInfo>& waste_vk_image_infos,
        std::vector<VkDescriptorBufferInfo>& waste_vk_buffer_infos,
        std::vector<VkBufferView>& waste_vk_texel_buffer_views
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkCopyDescriptorSet.html
    struct CopyDescriptorSet
    {
        DescriptorSetWPtr src_set;
        uint32_t src_binding;
        uint32_t src_array_element;
        DescriptorSetWPtr dst_set;
        uint32_t dst_binding;
        uint32_t dst_array_element;
        uint32_t descriptor_count;
    };

    VkCopyDescriptorSet CopyDescriptorSet_to_vk(const CopyDescriptorSet& copy);

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkBufferViewCreateInfo.html
    struct BufferViewConfig
    {
        VkFormat format;
        VkDeviceSize offset;
        VkDeviceSize range;
    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFormatProperties.html
    struct FormatProperties
    {
        VkFormatFeatureFlags linear_tiling_features;
        VkFormatFeatureFlags optimal_tiling_features;
        VkFormatFeatureFlags buffer_features;
    };

    FormatProperties FormatProperties_from_vk(
        const VkFormatProperties& properties
    );

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageFormatProperties.html
    struct ImageFormatProperties
    {
        Extent3d max_extent;
        uint32_t max_mip_levels;
        uint32_t max_array_layers;
        VkSampleCountFlags sample_counts;
        VkDeviceSize max_resource_size;
    };

    ImageFormatProperties ImageFormatProperties_from_vk(
        const VkImageFormatProperties& properties
    );

#pragma endregion

#pragma region error handling

    class Error
    {
    public:
        Error();

        Error(std::string message);

        Error(const VkResult vk_result);

        Error(
            std::string message,
            const std::optional<VkResult>& vk_result,
            bool vk_result_already_embedded_in_message
        );

        constexpr const std::optional<VkResult>& vk_result() const
        {
            return _vk_result;
        }

        std::string to_string() const;

    private:
        std::string message;
        std::optional<VkResult> _vk_result;
        bool stringify_vk_result;

    };

#pragma endregion

#pragma region classes and object wrappers

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkAllocationCallbacks.html
    class Allocator
    {
    public:
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/PFN_vkAllocationFunction.html
        virtual void* allocate(
            size_t size,
            size_t alignment,
            VkSystemAllocationScope allocation_scope
        ) = 0;

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/PFN_vkReallocationFunction.html
        virtual void* reallocate(
            void* original,
            size_t size,
            size_t alignment,
            VkSystemAllocationScope allocation_scope
        ) = 0;

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/PFN_vkFreeFunction.html
        virtual void free(void* memory) = 0;

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/PFN_vkInternalAllocationNotification.html
        virtual void internal_allocation_notification(
            size_t size,
            VkInternalAllocationType allocation_type,
            VkSystemAllocationScope allocation_scope
        ) = 0;

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/PFN_vkInternalFreeNotification.html
        virtual void internal_free_notification(
            size_t size,
            VkInternalAllocationType allocation_type,
            VkSystemAllocationScope allocation_scope
        ) = 0;

    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDevice.html
    class PhysicalDevice
    {
    public:
        constexpr VkPhysicalDevice handle() const
        {
            return _handle;
        }

        constexpr const PhysicalDeviceProperties& properties() const
        {
            return _properties;
        }

        constexpr const PhysicalDeviceFeatures& features() const
        {
            return _features;
        }

        constexpr const PhysicalDeviceMemoryProperties&
            memory_properties() const
        {
            return _memory_properties;
        }

        constexpr const std::vector<QueueFamily>& queue_families() const
        {
            return _queue_families;
        }

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkEnumerateDeviceExtensionProperties.html
        std::vector<ExtensionProperties> fetch_available_extensions(
            const std::string& layer_name = ""
        ) const;

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetPhysicalDeviceFormatProperties.html
        FormatProperties fetch_format_properties(VkFormat format) const;

        // find the first image format in the candidates that is supported with
        // the provided tiling and required features.
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetPhysicalDeviceFormatProperties.html
        std::optional<VkFormat> find_supported_image_format(
            const std::vector<VkFormat>& candidates,
            VkImageTiling tiling,
            VkFormatFeatureFlags features
        ) const;

        // might throw error with VK_ERROR_FORMAT_NOT_SUPPORTED
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetPhysicalDeviceImageFormatProperties.html
        ImageFormatProperties fetch_image_format_properties(
            VkFormat format,
            VkImageType type,
            VkImageTiling tiling,
            VkImageUsageFlags usage,
            VkImageCreateFlags flags
        ) const;

        // this will only have a value if the VK_KHR_swapchain extension is
        // available and surface is not nullptr.
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetPhysicalDeviceSurfaceCapabilitiesKHR.html
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetPhysicalDeviceSurfaceFormatsKHR.html
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetPhysicalDeviceSurfacePresentModesKHR.html
        std::optional<SwapchainSupport> fetch_swapchain_support(
            const SurfacePtr& surface
        ) const;

        // find indices of queue families that meet the provided criteria
        // surface support: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetPhysicalDeviceSurfaceSupportKHR.html
        std::vector<uint32_t> find_queue_family_indices(
            VkQueueFlags must_support,
            VkQueueFlags must_not_support = 0,
            const SurfacePtr& must_support_surface = nullptr,
            uint32_t min_queue_count = 1
        ) const;

        // find index of the first queue family that meets the provided
        // criteria. throws Error if not found.
        uint32_t find_first_queue_family_index(
            VkQueueFlags must_support,
            VkQueueFlags must_not_support = 0,
            const SurfacePtr& must_support_surface = nullptr,
            uint32_t min_queue_count = 1
        ) const;

    protected:
        VkPhysicalDevice _handle = nullptr;

        PhysicalDeviceProperties _properties;
        PhysicalDeviceFeatures _features;
        PhysicalDeviceMemoryProperties _memory_properties;
        std::vector<QueueFamily> _queue_families;

        PhysicalDevice(
            VkPhysicalDevice handle,
            const PhysicalDeviceProperties& properties,
            const PhysicalDeviceFeatures& features,
            const PhysicalDeviceMemoryProperties& memory_properties,
            const std::vector<QueueFamily>& queue_families
        );

        friend class Context;

    };

    // manages a VkInstance and custom allocators, provides utility functions,
    // and is used by other classes
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkInstance.html
    class Context
    {
    public:
        _BV_DELETE_DEFAULT_CTOR(Context);

        Context(const Context& other) = delete;
        Context& operator=(const Context& other) = delete;
        Context& operator=(Context&& other) = default;

        Context(Context&& other) noexcept;

        // it's best to keep at least one external reference to the allocator so
        // that it doesn't die with the Context because the driver might still
        // use the allocator even after the instance is destroyed.
        static ContextPtr create(
            const ContextConfig& config,
            const AllocatorPtr& allocator = nullptr
        );

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkEnumerateInstanceLayerProperties.html
        static std::vector<LayerProperties> fetch_available_layers();

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkEnumerateInstanceExtensionProperties.html
        static std::vector<ExtensionProperties>
            fetch_available_extensions(
                const std::string& layer_name = ""
            );

        constexpr const ContextConfig& config() const
        {
            return _config;
        }

        constexpr const AllocatorPtr& allocator() const
        {
            return _allocator;
        }

        // it's best to keep at least one external reference to the allocator so
        // that it doesn't die with the Context because the driver might still
        // use the allocator even after the instance is destroyed.
        void set_allocator(
            const AllocatorPtr& allocator
        );

        constexpr VkInstance vk_instance() const
        {
            return _vk_instance;
        }

        // if _allocator == nullptr, this will return nullptr, otherwise it will
        // return the address of _vk_allocator which contains function pointers
        // for callbacks.
        const VkAllocationCallbacks* vk_allocator_ptr() const;

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkEnumeratePhysicalDevices.html
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetPhysicalDeviceProperties.html
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetPhysicalDeviceFeatures.html
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetPhysicalDeviceMemoryProperties.html
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetPhysicalDeviceQueueFamilyProperties.html
        std::vector<PhysicalDevice> fetch_physical_devices() const;

        ~Context();

    protected:
        ContextConfig _config;

        AllocatorPtr _allocator;
        VkAllocationCallbacks _vk_allocator{};

        VkInstance _vk_instance = nullptr;

        Context(
            const ContextConfig& config,
            const AllocatorPtr& allocator
        );

    };

    // provided by VK_EXT_debug_utils
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDebugUtilsMessengerEXT.html
    class DebugMessenger
    {
    public:
        _BV_DELETE_DEFAULT_CTOR_AND_ALLOW_MOVE_ONLY(DebugMessenger);

        static DebugMessengerPtr create(
            const ContextPtr& context,
            VkDebugUtilsMessageSeverityFlagsEXT message_severity_filter,
            VkDebugUtilsMessageTypeFlagsEXT message_type_filter,
            const DebugCallback& callback
        );

        constexpr const ContextWPtr& context() const
        {
            return _context;
        }

        constexpr const VkDebugUtilsMessageSeverityFlagsEXT&
            message_severity_filter() const
        {
            return _message_severity_filter;
        }

        constexpr const VkDebugUtilsMessageTypeFlagsEXT&
            message_type_filter() const
        {
            return _message_type_filter;
        }

        constexpr const DebugCallback& callback() const
        {
            return _callback;
        }

        constexpr VkDebugUtilsMessengerEXT handle() const
        {
            return _handle;
        }

        ~DebugMessenger();

    protected:
        ContextWPtr _context;
        VkDebugUtilsMessageSeverityFlagsEXT _message_severity_filter;
        VkDebugUtilsMessageTypeFlagsEXT _message_type_filter;
        DebugCallback _callback;

        VkDebugUtilsMessengerEXT _handle = nullptr;

        DebugMessenger(
            const ContextPtr& context,
            VkDebugUtilsMessageSeverityFlagsEXT message_severity_filter,
            VkDebugUtilsMessageTypeFlagsEXT message_type_filter,
            const DebugCallback& callback
        );

    };

    // provided by VK_KHR_surface
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSurfaceKHR.html
    class Surface
    {
    public:
        _BV_DELETE_DEFAULT_CTOR_AND_ALLOW_MOVE_ONLY(Surface);

        // create a surface based on a user-provided handle. this lets the user
        // write their own little surface creation implementation based on the
        // platform or the windowing library they're using.
        // * make sure to enable the required extensions for surfaces. some
        //   windowing libraries (like GLFW) provide the list for you.
        static SurfacePtr create(
            const ContextPtr& context,
            VkSurfaceKHR handle
        );

        constexpr const ContextWPtr& context() const
        {
            return _context;
        }

        constexpr VkSurfaceKHR handle() const
        {
            return _handle;
        }

        ~Surface();

    protected:
        ContextWPtr _context;

        VkSurfaceKHR _handle;

        Surface(
            const ContextPtr& context,
            VkSurfaceKHR handle
        );

    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkQueue.html
    class Queue
    {
    public:
        _BV_DELETE_DEFAULT_CTOR_AND_ALLOW_MOVE_ONLY(Queue);

        constexpr const DeviceWPtr& device() const
        {
            return _device;
        }

        constexpr uint32_t queue_family_index() const
        {
            return _queue_family_index;
        }

        constexpr uint32_t queue_index() const
        {
            return _queue_index;
        }

        constexpr VkQueue handle() const
        {
            return _handle;
        }

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSubmitInfo.html
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkQueueSubmit.html
        void submit(
            const std::vector<VkPipelineStageFlags>& wait_stages,
            const std::vector<SemaphorePtr>& wait_semaphores,
            const std::vector<CommandBufferPtr>& command_buffers,
            const std::vector<SemaphorePtr>& signal_semaphores,
            const FencePtr& signal_fence = nullptr
        );

        // provided by VK_KHR_swapchain
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPresentInfoKHR.html
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkQueuePresentKHR.html
        void present(
            const std::vector<SemaphorePtr>& wait_semaphores,
            const SwapchainPtr& swapchain,
            uint32_t image_index,
            VkResult* out_vk_result = nullptr
        );

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkQueueWaitIdle.html
        void wait_idle();

    protected:
        DeviceWPtr _device;
        uint32_t _queue_family_index;
        uint32_t _queue_index;
        VkQueue _handle;

        Queue(
            const DeviceWPtr& device,
            uint32_t queue_family_index,
            uint32_t queue_index,
            VkQueue handle
        );

    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDevice.html
    class Device
    {
    public:
        _BV_DELETE_DEFAULT_CTOR_AND_ALLOW_MOVE_ONLY(Device);

        static DevicePtr create(
            const ContextPtr& context,
            const PhysicalDevice& physical_device,
            const DeviceConfig& config
        );

        constexpr const ContextWPtr& context() const
        {
            return _context;
        }

        constexpr const PhysicalDevice& physical_device() const
        {
            return _physical_device;
        }

        constexpr const DeviceConfig& config() const
        {
            return _config;
        }

        constexpr VkDevice handle() const
        {
            return _handle;
        }

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetDeviceQueue.html
        static QueuePtr retrieve_queue(
            const DevicePtr& device,
            uint32_t queue_family_index,
            uint32_t queue_index
        );

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkDeviceWaitIdle.html
        void wait_idle();

        ~Device();

    protected:
        ContextWPtr _context;
        PhysicalDevice _physical_device;
        DeviceConfig _config;

        VkDevice _handle = nullptr;

        Device(
            const ContextPtr& context,
            const PhysicalDevice& physical_device,
            const DeviceConfig& config
        );

    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImage.html
    class Image
    {
    public:
        _BV_DELETE_DEFAULT_CTOR_AND_ALLOW_MOVE_ONLY(Image);

        // this will make sure the requested format is supported with the
        // provided parameters. it will also fetch and store the memory
        // requirements.
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetPhysicalDeviceImageFormatProperties.html
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCreateImage.html
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetImageMemoryRequirements.html
        static ImagePtr create(
            const DevicePtr& device,
            const ImageConfig& config
        );

        // if true, then device(), config(), and memory_requirements() will
        // have useless default values (nullptr and zeros) and you shouldn't
        // use bind_memory() either, or anything other than handle(). this is
        // for swapchain images which aren't created by the user, so we have no
        // information about them but their handle.
        constexpr bool created_externally() const
        {
            return _created_externally;
        }

        constexpr const DeviceWPtr& device() const
        {
            return _device;
        }

        constexpr const ImageConfig& config() const
        {
            return _config;
        }

        constexpr const MemoryRequirements& memory_requirements() const
        {
            return _memory_requirements;
        }

        constexpr VkImage handle() const
        {
            return _handle;
        }

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkBindImageMemory.html
        void bind_memory(
            const DeviceMemoryPtr& memory,
            VkDeviceSize memory_offset
        );

        ~Image();

    protected:
        bool _created_externally;

        DeviceWPtr _device;
        ImageConfig _config;

        MemoryRequirements _memory_requirements{};

        VkImage _handle;

        Image(
            const DevicePtr& device,
            const ImageConfig& config
        );

        // this should only be used by Swapchain when retrieving its images
        Image(VkImage handle_created_externally);

    };

    // provided by VK_KHR_swapchain
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSwapchainKHR.html
    class Swapchain
    {
    public:
        _BV_DELETE_DEFAULT_CTOR_AND_ALLOW_MOVE_ONLY(Swapchain);

        // this will fetch and store the swapchain images automatically
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCreateSwapchainKHR.html
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetSwapchainImagesKHR.html
        static SwapchainPtr create(
            const DevicePtr& device,
            const SurfacePtr& surface,
            const SwapchainConfig& config,
            const SwapchainPtr& old_swapchain = nullptr
        );

        constexpr const DeviceWPtr& device() const
        {
            return _device;
        }

        constexpr const SurfaceWPtr& surface() const
        {
            return _surface;
        }

        constexpr const SwapchainConfig& config() const
        {
            return _config;
        }

        constexpr const std::optional<SwapchainWPtr>& old_swapchain() const
        {
            return _old_swapchain;
        }

        const std::vector<ImagePtr>& images() const
        {
            return _images;
        }

        constexpr VkSwapchainKHR handle() const
        {
            return _handle;
        }

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkAcquireNextImageKHR.html
        uint32_t acquire_next_image(
            const SemaphorePtr& semaphore = nullptr,
            const FencePtr& fence = nullptr,
            uint64_t timeout = std::numeric_limits<uint64_t>::max(),
            VkResult* out_vk_result = nullptr
        );

        ~Swapchain();

    protected:
        DeviceWPtr _device;
        SurfaceWPtr _surface;
        SwapchainConfig _config;
        std::optional<SwapchainWPtr> _old_swapchain;

        VkSwapchainKHR _handle = nullptr;

        std::vector<ImagePtr> _images;

        Swapchain(
            const DevicePtr& device,
            const SurfacePtr& surface,
            const SwapchainConfig& config,
            const SwapchainPtr& old_swapchain = nullptr
        );

    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageView.html
    class ImageView
    {
    public:
        _BV_DELETE_DEFAULT_CTOR_AND_ALLOW_MOVE_ONLY(ImageView);

        static ImageViewPtr create(
            const DevicePtr& device,
            const ImagePtr& image,
            const ImageViewConfig& config
        );

        constexpr const DeviceWPtr& device() const
        {
            return _device;
        }

        constexpr const ImageWPtr& image() const
        {
            return _image;
        }

        constexpr const ImageViewConfig& config() const
        {
            return _config;
        }

        constexpr VkImageView handle() const
        {
            return _handle;
        }

        ~ImageView();

    protected:
        DeviceWPtr _device;
        ImageWPtr _image;
        ImageViewConfig _config;

        VkImageView _handle = nullptr;

        ImageView(
            const DevicePtr& device,
            const ImagePtr& image,
            const ImageViewConfig& config
        );

    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkShaderModule.html
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkShaderModuleCreateInfo.html
    class ShaderModule
    {
    public:
        _BV_DELETE_DEFAULT_CTOR_AND_ALLOW_MOVE_ONLY(ShaderModule);

        static ShaderModulePtr create(
            const DevicePtr& device,
            const std::vector<uint8_t>& code
        );

        constexpr const DeviceWPtr& device() const
        {
            return _device;
        }

        constexpr VkShaderModule handle() const
        {
            return _handle;
        }

        ~ShaderModule();

    protected:
        DeviceWPtr _device;

        VkShaderModule _handle = nullptr;

        ShaderModule(const DevicePtr& device);

    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSampler.html
    class Sampler
    {
    public:
        _BV_DELETE_DEFAULT_CTOR_AND_ALLOW_MOVE_ONLY(Sampler);

        static SamplerPtr create(
            const DevicePtr& device,
            const SamplerConfig& config
        );

        constexpr const DeviceWPtr& device() const
        {
            return _device;
        }

        constexpr const SamplerConfig& config() const
        {
            return _config;
        }

        constexpr VkSampler handle() const
        {
            return _handle;
        }

        ~Sampler();

    protected:
        DeviceWPtr _device;
        SamplerConfig _config;

        VkSampler _handle = nullptr;

        Sampler(
            const DevicePtr& device,
            const SamplerConfig& config
        );

    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDescriptorSetLayout.html
    class DescriptorSetLayout
    {
    public:
        _BV_DELETE_DEFAULT_CTOR_AND_ALLOW_MOVE_ONLY(DescriptorSetLayout);

        static DescriptorSetLayoutPtr create(
            const DevicePtr& device,
            const DescriptorSetLayoutConfig& config
        );

        constexpr const DeviceWPtr& device() const
        {
            return _device;
        }

        constexpr const DescriptorSetLayoutConfig& config() const
        {
            return _config;
        }

        constexpr VkDescriptorSetLayout handle() const
        {
            return _handle;
        }

        ~DescriptorSetLayout();

    protected:
        DeviceWPtr _device;
        DescriptorSetLayoutConfig _config;

        VkDescriptorSetLayout _handle = nullptr;

        DescriptorSetLayout(
            const DevicePtr& device,
            const DescriptorSetLayoutConfig& config
        );

    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineLayout.html
    class PipelineLayout
    {
    public:
        _BV_DELETE_DEFAULT_CTOR_AND_ALLOW_MOVE_ONLY(PipelineLayout);

        static PipelineLayoutPtr create(
            const DevicePtr& device,
            const PipelineLayoutConfig& config
        );

        constexpr const DeviceWPtr& device() const
        {
            return _device;
        }

        constexpr const PipelineLayoutConfig& config() const
        {
            return _config;
        }

        constexpr VkPipelineLayout handle() const
        {
            return _handle;
        }

        ~PipelineLayout();

    protected:
        DeviceWPtr _device;
        PipelineLayoutConfig _config;

        VkPipelineLayout _handle = nullptr;

        PipelineLayout(
            const DevicePtr& device,
            const PipelineLayoutConfig& config
        );

    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkRenderPass.html
    class RenderPass
    {
    public:
        _BV_DELETE_DEFAULT_CTOR_AND_ALLOW_MOVE_ONLY(RenderPass);

        static RenderPassPtr create(
            const DevicePtr& device,
            const RenderPassConfig& config
        );

        constexpr const DeviceWPtr& device() const
        {
            return _device;
        }

        constexpr const RenderPassConfig& config() const
        {
            return _config;
        }

        constexpr VkRenderPass handle() const
        {
            return _handle;
        }

        ~RenderPass();

    protected:
        DeviceWPtr _device;
        RenderPassConfig _config;

        VkRenderPass _handle = nullptr;

        RenderPass(
            const DevicePtr& device,
            const RenderPassConfig& config
        );

    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipeline.html
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCreateGraphicsPipelines.html
    class GraphicsPipeline
    {
    public:
        _BV_DELETE_DEFAULT_CTOR_AND_ALLOW_MOVE_ONLY(GraphicsPipeline);

        static GraphicsPipelinePtr create(
            const DevicePtr& device,
            const GraphicsPipelineConfig& config,
            const PipelineCachePtr& cache = nullptr
        );

        constexpr const DeviceWPtr& device() const
        {
            return _device;
        }

        constexpr const GraphicsPipelineConfig& config() const
        {
            return _config;
        }

        constexpr const PipelineCacheWPtr& cache() const
        {
            return _cache;
        }

        constexpr VkPipeline handle() const
        {
            return _handle;
        }

        ~GraphicsPipeline();

    protected:
        DeviceWPtr _device;
        GraphicsPipelineConfig _config;
        PipelineCacheWPtr _cache;

        VkPipeline _handle = nullptr;

        GraphicsPipeline(
            const DevicePtr& device,
            const GraphicsPipelineConfig& config,
            const PipelineCachePtr& cache = nullptr
        );

    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipeline.html
    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCreateComputePipelines.html
    class ComputePipeline
    {
    public:
        _BV_DELETE_DEFAULT_CTOR_AND_ALLOW_MOVE_ONLY(ComputePipeline);

        static ComputePipelinePtr create(
            const DevicePtr& device,
            const ComputePipelineConfig& config,
            const PipelineCachePtr& cache = nullptr
        );

        constexpr const DeviceWPtr& device() const
        {
            return _device;
        }

        constexpr const ComputePipelineConfig& config() const
        {
            return _config;
        }

        constexpr const PipelineCacheWPtr& cache() const
        {
            return _cache;
        }

        constexpr VkPipeline handle() const
        {
            return _handle;
        }

        ~ComputePipeline();

    protected:
        DeviceWPtr _device;
        ComputePipelineConfig _config;
        PipelineCacheWPtr _cache;

        VkPipeline _handle = nullptr;

        ComputePipeline(
            const DevicePtr& device,
            const ComputePipelineConfig& config,
            const PipelineCachePtr& cache = nullptr
        );

    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebuffer.html
    class Framebuffer
    {
    public:
        _BV_DELETE_DEFAULT_CTOR_AND_ALLOW_MOVE_ONLY(Framebuffer);

        static FramebufferPtr create(
            const DevicePtr& device,
            const FramebufferConfig& config
        );

        constexpr const DeviceWPtr& device() const
        {
            return _device;
        }

        constexpr const FramebufferConfig& config() const
        {
            return _config;
        }

        constexpr VkFramebuffer handle() const
        {
            return _handle;
        }

        ~Framebuffer();

    protected:
        DeviceWPtr _device;
        FramebufferConfig _config;

        VkFramebuffer _handle = nullptr;

        Framebuffer(
            const DevicePtr& device,
            const FramebufferConfig& config
        );

    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkCommandBuffer.html
    class CommandBuffer
    {
    public:
        _BV_DELETE_DEFAULT_CTOR_AND_ALLOW_MOVE_ONLY(CommandBuffer);

        constexpr const CommandPoolWPtr& pool() const
        {
            return _pool;
        }

        constexpr VkCommandBuffer handle() const
        {
            return _handle;
        }

        void reset(VkCommandBufferResetFlags flags);

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkCommandBufferBeginInfo.html
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkBeginCommandBuffer.html
        void begin(
            VkCommandBufferUsageFlags flags,
            std::optional<CommandBufferInheritance> inheritance = std::nullopt
        );

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkEndCommandBuffer.html
        void end();

        ~CommandBuffer();

    protected:
        CommandPoolWPtr _pool;
        VkCommandBuffer _handle;

        CommandBuffer(
            const CommandPoolWPtr& pool,
            VkCommandBuffer handle
        );

    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkCommandPool.html
    class CommandPool
    {
    public:
        _BV_DELETE_DEFAULT_CTOR_AND_ALLOW_MOVE_ONLY(CommandPool);

        static CommandPoolPtr create(
            const DevicePtr& device,
            const CommandPoolConfig& config
        );

        constexpr const DeviceWPtr& device() const
        {
            return _device;
        }

        constexpr const CommandPoolConfig& config() const
        {
            return _config;
        }

        constexpr VkCommandPool handle() const
        {
            return _handle;
        }

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkCommandBufferAllocateInfo.html
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkAllocateCommandBuffers.html
        static CommandBufferPtr allocate_buffer(
            const CommandPoolPtr& pool,
            VkCommandBufferLevel level
        );

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkCommandBufferAllocateInfo.html
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkAllocateCommandBuffers.html
        static std::vector<CommandBufferPtr> allocate_buffers(
            const CommandPoolPtr& pool,
            VkCommandBufferLevel level,
            uint32_t count
        );

        ~CommandPool();

    protected:
        DeviceWPtr _device;
        CommandPoolConfig _config;

        VkCommandPool _handle = nullptr;

        CommandPool(
            const DevicePtr& device,
            const CommandPoolConfig& config
        );

    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSemaphore.html
    class Semaphore
    {
    public:
        _BV_DELETE_DEFAULT_CTOR_AND_ALLOW_MOVE_ONLY(Semaphore);

        static SemaphorePtr create(const DevicePtr& device);

        constexpr const DeviceWPtr& device() const
        {
            return _device;
        }

        constexpr VkSemaphore handle() const
        {
            return _handle;
        }

        ~Semaphore();

    protected:
        DeviceWPtr _device;

        VkSemaphore _handle = nullptr;

        Semaphore(const DevicePtr& device);

    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFence.html
    class Fence
    {
    public:
        _BV_DELETE_DEFAULT_CTOR_AND_ALLOW_MOVE_ONLY(Fence);

        static FencePtr create(
            const DevicePtr& device,
            VkFenceCreateFlags flags
        );

        constexpr const DeviceWPtr& device() const
        {
            return _device;
        }

        constexpr VkFence handle() const
        {
            return _handle;
        }

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkWaitForFences.html
        void wait(uint64_t timeout = UINT64_MAX);

        // all provided fences must be from the same device, bad things might
        // happen otherwise.
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkWaitForFences.html
        static void wait_multiple(
            const std::vector<FencePtr>& fences,
            bool wait_all,
            uint64_t timeout = UINT64_MAX
        );

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkResetFences.html
        void reset();

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetFenceStatus.html
        bool is_signaled() const;

        ~Fence();

    protected:
        DeviceWPtr _device;

        VkFence _handle = nullptr;

        Fence(const DevicePtr& device);

    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkBuffer.html
    class Buffer
    {
    public:
        _BV_DELETE_DEFAULT_CTOR_AND_ALLOW_MOVE_ONLY(Buffer);

        // this will automatically fetch and store the memory requirements
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCreateBuffer.html
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetBufferMemoryRequirements.html
        static BufferPtr create(
            const DevicePtr& device,
            const BufferConfig& config
        );

        constexpr const DeviceWPtr& device() const
        {
            return _device;
        }

        constexpr const BufferConfig& config() const
        {
            return _config;
        }

        constexpr const MemoryRequirements& memory_requirements() const
        {
            return _memory_requirements;
        }

        constexpr VkBuffer handle() const
        {
            return _handle;
        }

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkBindBufferMemory.html
        void bind_memory(
            const DeviceMemoryPtr& memory,
            VkDeviceSize memory_offset
        );

        ~Buffer();

    protected:
        DeviceWPtr _device;
        BufferConfig _config;

        MemoryRequirements _memory_requirements{};

        VkBuffer _handle = nullptr;

        Buffer(
            const DevicePtr& device,
            const BufferConfig& config
        );

    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDeviceMemory.html
    class DeviceMemory
    {
    public:
        _BV_DELETE_DEFAULT_CTOR_AND_ALLOW_MOVE_ONLY(DeviceMemory);

        static DeviceMemoryPtr allocate(
            const DevicePtr& device,
            const DeviceMemoryConfig& config
        );

        constexpr const DeviceWPtr& device() const
        {
            return _device;
        }

        constexpr const DeviceMemoryConfig& config() const
        {
            return _config;
        }

        constexpr VkDeviceMemory handle() const
        {
            return _handle;
        }

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkMapMemory.html
        void* map(VkDeviceSize offset, VkDeviceSize size);

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkUnmapMemory.html
        void unmap();

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkFlushMappedMemoryRanges.html
        void flush_mapped_range(VkDeviceSize offset, VkDeviceSize size);

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkInvalidateMappedMemoryRanges.html
        void invalidate_mapped_range(
            VkDeviceSize offset,
            VkDeviceSize size
        );

        // map the whole memory, copy the provided data, flush, and unmap.
        // you should make sure that mapping is allowed for this memory.
        void upload(void* data, VkDeviceSize data_size);

        ~DeviceMemory();

    protected:
        DeviceWPtr _device;
        DeviceMemoryConfig _config;

        VkDeviceMemory _handle = nullptr;

        DeviceMemory(
            const DevicePtr& device,
            const DeviceMemoryConfig& config
        );

    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDescriptorSet.html
    class DescriptorSet
    {
    public:
        _BV_DELETE_DEFAULT_CTOR_AND_ALLOW_MOVE_ONLY(DescriptorSet);

        constexpr const DescriptorPoolWPtr& pool() const
        {
            return _pool;
        }

        constexpr VkDescriptorSet handle() const
        {
            return _handle;
        }

        // all provided sets and nested objects must be from the provided
        // device, bad things might happen otherwise.
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkUpdateDescriptorSets.html
        static void update_sets(
            const DevicePtr& device,
            const std::vector<WriteDescriptorSet>& writes,
            const std::vector<CopyDescriptorSet>& copies
        );

        ~DescriptorSet();

    protected:
        DescriptorPoolWPtr _pool;
        VkDescriptorSet _handle;

        DescriptorSet(
            const DescriptorPoolPtr& pool,
            VkDescriptorSet handle
        );

    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDescriptorPool.html
    class DescriptorPool
    {
    public:
        _BV_DELETE_DEFAULT_CTOR_AND_ALLOW_MOVE_ONLY(DescriptorPool);

        static DescriptorPoolPtr create(
            const DevicePtr& device,
            const DescriptorPoolConfig& config
        );

        constexpr const DeviceWPtr& device() const
        {
            return _device;
        }

        constexpr const DescriptorPoolConfig& config() const
        {
            return _config;
        }

        constexpr VkDescriptorPool handle() const
        {
            return _handle;
        }

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDescriptorSetAllocateInfo.html
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkAllocateDescriptorSets.html
        static DescriptorSetPtr allocate_set(
            const DescriptorPoolPtr& pool,
            const DescriptorSetLayoutPtr& set_layout
        );

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkDescriptorSetAllocateInfo.html
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkAllocateDescriptorSets.html
        static std::vector<DescriptorSetPtr> allocate_sets(
            const DescriptorPoolPtr& pool,
            uint32_t count,
            const std::vector<DescriptorSetLayoutPtr>& set_layouts
        );

        ~DescriptorPool();

    protected:
        DeviceWPtr _device;
        DescriptorPoolConfig _config;

        VkDescriptorPool _handle = nullptr;

        DescriptorPool(
            const DevicePtr& device,
            const DescriptorPoolConfig& config
        );

    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkBufferView.html
    class BufferView
    {
    public:
        _BV_DELETE_DEFAULT_CTOR_AND_ALLOW_MOVE_ONLY(BufferView);

        static BufferViewPtr create(
            const DevicePtr& device,
            const BufferPtr& buffer,
            const BufferViewConfig& config
        );

        constexpr const DeviceWPtr& device() const
        {
            return _device;
        }

        constexpr const BufferWPtr& buffer() const
        {
            return _buffer;
        }

        constexpr const BufferViewConfig& config() const
        {
            return _config;
        }

        constexpr VkBufferView handle() const
        {
            return _handle;
        }

        ~BufferView();

    protected:
        DeviceWPtr _device;
        BufferWPtr _buffer;
        BufferViewConfig _config;

        VkBufferView _handle = nullptr;

        BufferView(
            const DevicePtr& device,
            const BufferPtr& buffer,
            const BufferViewConfig& config
        );

    };

    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineCache.html
    class PipelineCache
    {
    public:
        _BV_DELETE_DEFAULT_CTOR_AND_ALLOW_MOVE_ONLY(PipelineCache);

        static PipelineCachePtr create(
            const DevicePtr& device,
            VkPipelineCacheCreateFlags flags,
            const std::vector<uint8_t>& initial_data
        );

        constexpr const DeviceWPtr& device() const
        {
            return _device;
        }

        constexpr VkPipelineCacheCreateFlags flags() const
        {
            return _flags;
        }

        constexpr VkPipelineCache handle() const
        {
            return _handle;
        }

        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkGetPipelineCacheData.html
        std::vector<uint8_t> get_cache_data();

        ~PipelineCache();

    protected:
        DeviceWPtr _device;
        VkPipelineCacheCreateFlags _flags;

        VkPipelineCache _handle = nullptr;

        PipelineCache(
            const DevicePtr& device,
            VkPipelineCacheCreateFlags flags
        );

    };

#pragma endregion

#pragma region helper functions

    // zero out a vector's capacity to actually free the memory
    template<typename T>
    void clear(std::vector<T>& vec)
    {
        vec.clear();
        std::vector<T>().swap(vec);
    }

    template<size_t size, typename T>
    std::array<T, size> raw_arr_to_std(const T* raw_arr)
    {
        std::array<T, size> arr;
        std::copy(raw_arr, raw_arr + size, arr.data());
        return arr;
    }

    std::string cstr_to_std(const char* cstr);

    // try to lock a weak pointer and throw an Error if it has expired
    template<typename T>
    std::shared_ptr<T> lock_wptr(const std::weak_ptr<T>& wptr)
    {
        if (wptr.expired())
        {
            throw Error("weak pointer has expired");
        }
        return wptr.lock();
    }

    bool format_has_depth_component(VkFormat format);
    bool format_has_stencil_component(VkFormat format);

#pragma endregion

#pragma region memory management

    class MemoryRegion
    {
    public:
        _BV_DELETE_DEFAULT_CTOR_AND_ALLOW_MOVE_ONLY(MemoryRegion);

    protected:
        bv::DeviceMemoryPtr mem;
        sul::dynamic_bitset<> blocks; // for each block: 0 = free, 1 = allocated
        void* mapped = nullptr;

        MemoryRegion(const bv::DeviceMemoryPtr& mem);

        friend class MemoryChunk;
        friend class MemoryBank;

    };

    class MemoryChunk
    {
    public:
        _BV_DELETE_DEFAULT_CTOR_AND_ALLOW_MOVE_ONLY(MemoryChunk);

        const bv::DeviceMemoryPtr& memory() const;

        constexpr VkDeviceSize offset() const
        {
            return _offset;
        }

        constexpr VkDeviceSize size() const
        {
            return _size;
        }

        void bind(bv::BufferPtr& buffer);
        void bind(bv::ImagePtr& image);

        void* mapped();
        void flush();

        ~MemoryChunk();

    protected:
        std::shared_ptr<std::mutex> mutex;
        MemoryRegionPtr region;
        VkDeviceSize _offset;
        VkDeviceSize _size;
        VkDeviceSize block_size;

        MemoryChunk(
            const std::shared_ptr<std::mutex>& mutex,
            const MemoryRegionPtr& region,
            VkDeviceSize offset,
            VkDeviceSize size,
            VkDeviceSize block_size
        );

    };

    class MemoryBank
    {
    public:
        _BV_DELETE_DEFAULT_CTOR_AND_ALLOW_MOVE_ONLY(MemoryBank);

        static MemoryBankPtr create(
            const DevicePtr& device,
            VkDeviceSize block_size = 1024,
            VkDeviceSize min_region_size = 268'435'456
        );

        constexpr const bv::DevicePtr& device() const
        {
            return _device;
        }

        constexpr VkDeviceSize block_size() const
        {
            return _block_size;
        }

        constexpr VkDeviceSize min_region_size() const
        {
            return _min_region_size;
        }

        MemoryChunkPtr allocate(
            const bv::MemoryRequirements& requirements,
            VkMemoryPropertyFlags required_properties
        );

        // returns a string description of its status including the regions
        std::string to_string();

        ~MemoryBank();

    protected:
        bv::DevicePtr _device;
        std::vector<MemoryRegionPtr> regions;
        std::shared_ptr<std::mutex> mutex;

        VkDeviceSize _block_size;
        VkDeviceSize _min_region_size;

        MemoryBank(
            const bv::DevicePtr& device,
            VkDeviceSize block_size,
            VkDeviceSize min_region_size
        );

        void delete_empty_regions();

    };

#pragma endregion

}
