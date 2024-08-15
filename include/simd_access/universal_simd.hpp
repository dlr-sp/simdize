// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_UNIVERSAL_SIMD
#define SIMD_ACCESS_UNIVERSAL_SIMD

#include "simd_access/base.hpp"
#include "simd_access/index.hpp"
#include "simd_access/reflection.hpp"

namespace simd_access
{

template<class T, int SimdSize>
struct universal_simd : std::array<T, SimdSize>
{
  static constexpr auto size() { return SimdSize; }

  template<class G>
  explicit universal_simd(G&& generator) :
    std::array<T, SimdSize>([&]<int... I>(std::integer_sequence<int, I...>) -> std::array<T, SimdSize>
      {
        return {{ generator(std::integral_constant<int, I>())... }};
      } (std::make_integer_sequence<int, SimdSize>())) {}

  universal_simd() = default;
};

inline decltype(auto) generate_universal(std::integral auto i, auto&& generator)
{
  return generator(i);
}

template<class IndexType>
  requires(!std::integral<IndexType>)
inline decltype(auto) generate_universal(const IndexType& idx, auto&& generator)
{
  return universal_simd<decltype(generator(0)), idx.size()>([&](auto i) { return generator(get_index(idx, i)); });
}

template<class T, class Func>
  requires(!is_simd<T>)
inline decltype(auto) universal_access(const T& v, Func&& subobject)
{
  return subobject(v);
}

template<class T, int SimdSize, class Func>
inline auto universal_access(const simd_access::universal_simd<T, SimdSize>& v, Func&& subobject)
{
  decltype(simdized_value<SimdSize>(std::declval<decltype(subobject(v[0]))>())) result;
  for (int i = 0; i < SimdSize; ++i)
  {
    simd_members(result, subobject(v[i]), [&](auto&& d, auto&& s) { d[i] = s; });
  }
  return result;
}


#define SIMD_UNIVERSAL_ACCESS(value, expr) \
  simd_access::universal_access(value, [&](auto element) { return element expr; })


} //namespace simd_access

#endif //SIMD_ACCESS_UNIVERSAL_SIMD
