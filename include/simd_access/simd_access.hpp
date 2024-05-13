// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Helper functions and a macro to mark a simd access.
 */

#ifndef SIMD_ACCESS_MAIN
#define SIMD_ACCESS_MAIN

#include <concepts>

#include "simd_access/base.hpp"
#include "simd_access/element_access.hpp"
#include "simd_access/index.hpp"
#include "simd_access/load_store.hpp"
#include "simd_access/member_overload.hpp"
#include "simd_access/value_access.hpp"

namespace simd_access
{

auto get_base_address(auto& base_addr, std::integral auto i)
{
  return &base_addr[i];
}

template<int SimdSize, class IndexType>
auto get_base_address(auto& base_addr, const index<SimdSize, IndexType>& i)
{
  return &base_addr[i.index_];
}

template<int SimdSize, class ArrayType>
auto get_base_address(auto& base_addr, const index_array<SimdSize, ArrayType>&)
{
  return &base_addr[0];
}

auto get_base_address(auto& base_addr, const is_stdx_simd auto& i)
{
  return &base_addr[0];
}

template<size_t ElementSize, class T, int SimdSize, class IndexType>
auto get_direct_value_access(T* base, const index<SimdSize, IndexType>&)
{
  return make_value_access<ElementSize>(linear_location<T, SimdSize>{base});
}

template<size_t ElementSize, class T, int SimdSize, class ArrayType>
auto get_direct_value_access(T* base, const index_array<SimdSize, ArrayType>& indices)
{
  return make_value_access<ElementSize>(indexed_location<T, SimdSize, ArrayType>{base, indices.index_});
}

template<size_t ElementSize>
auto& get_direct_value_access(auto* base, std::integral auto)
{
  return *base;
}

template<size_t ElementSize, class T, class IndexType, class Abi>
auto get_direct_value_access(T* base, const stdx::simd<IndexType, Abi>& i)
{
  return make_value_access<ElementSize>(indexed_location<T, i.size(), stdx::simd<IndexType, Abi>>{base, i});
}

} //namespace simd_access


/**
 * This macro defines a simd access to a variable of the form `base[index] member` (member is optional).
 * TODO: If global operator[] overloading becomes possible, then a decomposition of `base[index]` isn't required
 * anymore. Direct accesses (`base[index]`) and accesses to sub-arrays (`base[index][subindex]`) can be directly
 * written then.
 * TODO: If operator.() overloading becomes possible too, then this macro becomes obsolete, since all expressions can
 * be directly written.
 */
#define SIMD_ACCESS(base, index, ...) \
  simd_access::get_direct_value_access<sizeof(decltype(base[0]))>( \
    &((*simd_access::get_base_address(base, index)) __VA_ARGS__), index)

#endif //SIMD_ACCESS_MAIN
