// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <array>
#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

template<class T, int SimdSize>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

template<class PotentialSimdType>
concept is_stdx_simd =
  requires(PotentialSimdType x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept is_simd =
  is_stdx_simd<PotentialSimdType> ||
  requires(PotentialSimdType x) { []<class T, int SimdSize>(universal_simd<T, SimdSize>&){}(x); };


template<class T, int SimdSize>
struct auto_simd
{
  using type = universal_simd<T, SimdSize>;
};

template<simd_arithmetic T, int SimdSize>
struct auto_simd<T, SimdSize>
{
  using type = stdx::fixed_size_simd<T, SimdSize>;
};

template<class T, int SimdSize>
using auto_simd_t = typename auto_simd<T, SimdSize>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE
