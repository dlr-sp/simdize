// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Elementwise access of simd elements.
 */

#ifndef SIMD_ACCESS_ELEMENT_ACCESS
#define SIMD_ACCESS_ELEMENT_ACCESS

#include <concepts>
#include <experimental/simd>
namespace stdx = std::experimental;

namespace simd_access
{

template<class PotentialIndexType>
concept is_simd =
  requires(PotentialIndexType x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

/**
 * Call a function for each scalar of a simd value.
 * @param x Simd value.
 * @param fn Function called for each scalar value of `x`. If the function intends to write to the element, then it
 *   must forward its argument as in `[&](auto&& y) { element_write(y) = ...; }`.
 */
void elementwise(is_simd auto&& x, auto&& fn)
{
  for (int i = 0; i < x.size(); ++i)
  {
    fn(x[i]);
  }
}

/**
 * Call a function for the scalar value `x`.
 * This overload can be used with a generic functor to uniformly access scalar values of simds and scalars.
 * @param x Scalar value.
 * @param fn Function called for `x`.
 */
void elementwise(auto&& x, auto&& fn)
{
  fn(x);
}

/**
 * Used in functors for `elementwise`, which write to the elements. The function and its overload is used
 * to distinguish between usual value types and simd::reference types.
 * @param x Element value.
 * @return Reference to `x`.
 */
auto& element_write(std::copyable auto& x)
{
  return x;
}

/**
 * Overloaded function intended to be called for simd::reference types, which are not copyable.
 * @param T Type of the element value.
 * @param x Element value.
 * @return Rvalue reference to `x`, which can be used in assignment (since only
 *   `simd::reference::operator=(auto&&) &&` is available.
 */
template<class T>
std::remove_reference_t<T>&& element_write(T&& x)
{
  return static_cast<typename std::remove_reference<T>::type&&>(x) ;
}

} //namespace simd_access

#endif //SIMD_ACCESS_ELEMENT_ACCESS
