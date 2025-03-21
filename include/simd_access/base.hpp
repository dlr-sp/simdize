// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

/// Forward declaration
template<class T, int SimdSize>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<std::remove_cvref_t<T>> && !std::is_same_v<std::remove_cvref_t<T>, bool>;

// This works only without non-type template parameters. Hopefully there will be a universal solution (see p2989).
template<class TestClass, template<typename...> typename ClassTemplate>
concept specialization_of =
  requires(std::remove_cvref_t<TestClass> x) { []<typename... Args>(ClassTemplate<Args...>&){}(x); };

template<class PotentialSimdType>
concept stdx_simd =
  requires(std::remove_cvref_t<PotentialSimdType> x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept any_simd =
  stdx_simd<PotentialSimdType> ||
  requires(std::remove_cvref_t<PotentialSimdType> x) { []<class T, int SimdSize>(universal_simd<T, SimdSize>&){}(x); };

/// Helper class to auto-generate either a `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type.
 * @tparam SimdSize Requested simd size.
 */
template<class T, int SimdSize>
struct auto_simd
{
  /// Universal simd type.
  using type = universal_simd<T, SimdSize>;
};

/// Specialization of \ref auto_simd for the `stdx::simd` variant,
/**
 * @tparam T Arithmetic value type.
 * @tparam SimdSize Requested simd size.
 */
template<simd_arithmetic T, int SimdSize>
struct auto_simd<T, SimdSize>
{
  /// stdx::simd type.
  using type = stdx::fixed_size_simd<T, SimdSize>;
};

/// Type which resolves either to `stdx::simd` or - if not applicable - a \ref universal_simd.
/**
 * @tparam T Value type. If arithmetic, the resulting type is a `stdx::simd`. Otherwise it is a `universal_simd`.
 * @tparam SimdSize Requested simd size.
 */
template<class T, int SimdSize>
using auto_simd_t = typename auto_simd<T, SimdSize>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE
