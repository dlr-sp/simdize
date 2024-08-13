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
#include "simd_access/simd_loop.hpp"
#include "simd_access/reflection.hpp"
#include "simd_access/value_access.hpp"

namespace simd_access
{

template<bool isLvalue>
struct LValueSeparator;

template<>
struct LValueSeparator<true>
{
  template<class T>
  static decltype(auto) to_simd(T&& base, std::integral auto i)
  {
    return base[i];
  }

  template<class T, class Func>
  static decltype(auto) to_simd(T&& base, std::integral auto i, Func&& subobject)
  {
    return subobject(base[i]);
  }

  template<int SimdSize, class IndexType>
  static auto get_base_address(auto&& base_addr, const index<SimdSize, IndexType>& i)
  {
    return &base_addr[i.index_];
  }

  template<class IndexType, class Abi>
  static auto get_base_address(auto&& base_addr, const stdx::simd<IndexType, Abi>&)
  {
    return &base_addr[0];
  }

  template<int SimdSize, class ArrayType>
  static auto get_base_address(auto&& base_addr, const index_array<SimdSize, ArrayType>&)
  {
    return &base_addr[0];
  }

  template<int SimdSize, class IndexType>
  static auto get_base_address(auto&& base_addr, const index<SimdSize, IndexType>& i, auto&& subobject)
  {
    return &subobject(base_addr[i.index_]);
  }

  template<class IndexType, class Abi>
  static auto get_base_address(auto&& base_addr, const stdx::simd<IndexType, Abi>&, auto&& subobject)
  {
    return &subobject(base_addr[0]);
  }

  template<int SimdSize, class ArrayType>
  static auto get_base_address(auto&& base_addr, const index_array<SimdSize, ArrayType>&, auto&& subobject)
  {
    return &subobject(base_addr[0]);
  }


  template<size_t ElementSize, class T, int SimdSize, class IndexType>
  static auto get_direct_value_access(T* base, const index<SimdSize, IndexType>&)
  {
    return make_value_access<ElementSize>(linear_location<T, SimdSize>{base});
  }

  template<size_t ElementSize, class T, int SimdSize, class ArrayType>
  static auto get_direct_value_access(T* base, const index_array<SimdSize, ArrayType>& idx)
  {
    return make_value_access<ElementSize>(indexed_location<T, SimdSize, ArrayType>{base, idx.index_});
  }

  template<size_t ElementSize, class T, class IndexType, class Abi>
  static auto get_direct_value_access(T* base, const stdx::simd<IndexType, Abi>& idx)
  {
    return make_value_access<ElementSize>(indexed_location<T, idx.size(), stdx::simd<IndexType, Abi>>{base, idx});
  }

  template<class IndexType, class... Func>
    requires(!std::integral<IndexType>)
  static auto to_simd(auto&& base, const IndexType& indices, Func&&... subobject)
  {
    return get_direct_value_access<sizeof(decltype(base[0]))>(get_base_address(base, indices, subobject...), indices);
  }
};


template<>
struct LValueSeparator<false>
{
  static decltype(auto) to_simd(auto&& base, std::integral auto i)
  {
    return base[i];
  }

  static decltype(auto) to_simd(auto&& base, std::integral auto i, auto&& subobject)
  {
    return subobject(base[i]);
  }

  template<class T>
  using BaseType = std::decay_t<decltype(std::declval<T>()[0])>;

  template<class T, class Func>
  using BaseTypeFn = std::decay_t<decltype(std::declval<Func>()(std::declval<T>()[0]))>;

  template<class T, class IndexType>
    requires(!std::integral<IndexType>)
  static auto to_simd(T&& base, const IndexType& idx)
  {
    return load_rvalue<BaseType<T>>(base, idx);
  }

  template<class T, class IndexType, class Func>
    requires(!std::integral<IndexType>)
  static auto to_simd(T&& base, const IndexType& idx, Func&& subelement)
  {
    return load_rvalue<BaseTypeFn<T, Func>>(base, idx, subelement);
  }
};


/**
 * This function mocks a globally overloaded operator[]. It can be used instead of the SIMD_ACCESS macro,
 * if no named subobject is accessed.
 */
template<class T, class IndexType>
inline decltype(auto) sa(T&& base, const IndexType& index)
{
  return LValueSeparator<std::is_lvalue_reference_v<decltype(base[0])>>::to_simd(base, index);
}

template<has_to_simd T>
inline auto to_simd(const T& value)
{
  return value.to_simd();
}

template<class T> requires(!has_to_simd<T>)
inline auto to_simd(T&& value)
{
  return value;
}

} //namespace simd_access


/**
 * This macro defines a simd access to a variable of the form `base[index] subobject` (subobject is optional).
 * TODO: If global operator[] overloading becomes possible, then a decomposition of `base[index]` isn't required
 * anymore. Direct accesses (`base[index]`) and accesses to sub-arrays (`base[index][subindex]`) can be directly
 * written then.
 * TODO: If operator.() overloading becomes possible too, then this macro becomes obsolete, since all expressions can
 * be directly written.
 */
#define SIMD_ACCESS(base, index, ...) \
  simd_access::LValueSeparator<std::is_lvalue_reference_v<decltype((base[0] __VA_ARGS__))>>:: \
    to_simd(base, index __VA_OPT__(, [&](auto&& e) -> decltype((e __VA_ARGS__)) { return e __VA_ARGS__; }))

#define SIMD_ACCESS_V(...) simd_access::to_simd(SIMD_ACCESS(__VA_ARGS__))

#endif //SIMD_ACCESS_MAIN
