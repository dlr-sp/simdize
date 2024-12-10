// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Elementwise access of simd elements.
 */

#ifndef SIMD_ACCESS_ELEMENT_ACCESS
#define SIMD_ACCESS_ELEMENT_ACCESS

#include "simd_access/base.hpp"
#include "simd_access/index.hpp"
#include <utility>

namespace simd_access
{

template<class T>
concept is_simd_accessible = is_simd_index<T> || is_simd<T>;

inline auto element(const is_simd_index auto& index, int i)
{
  return get_index(index, i);
}

inline decltype(auto) element(is_simd auto&& value, int i)
{
  return value[i];
}

inline auto&& element(auto&& value)
{
  return value;
}

/**
 * Call a function for each scalar of a simd value.
 * @param fn Function called for each scalar value of `x` and `y`. If the function intends to write to the element,
 *   then it must forward its argument as in `[&](auto&& y) { element_write(y) = ...; }`.
 * @param x Simd value.
 * @param y... More simd values.
 */
inline void elementwise(auto&& fn, is_simd_accessible auto&& x, is_simd_accessible auto&&... y)
{
  for (int i = 0; i < x.size(); ++i)
  {
    fn(element(x, i), element(y, i)...);
  }
}

/**
 * Call a function for each scalar of a simd value.
 * @param fn Function called for each scalar value of `x` and `y`. If the function intends to write to the element,
 *   then it must forward its argument as in `[&](auto&& y) { element_write(y) = ...; }`.
 * @param x Simd value.
 * @param y... More simd values.
 */
inline void elementwise_with_index(auto&& fn, is_simd_accessible auto&& x, is_simd_accessible auto&&... y)
{
  for (int i = 0; i < x.size(); ++i)
  {
    fn(element(x, i), element(y, i)..., i);
  }
}

/**
 * Call a function for the scalar value `x`.
 * This overload can be used with a generic functor to uniformly access scalar values of simds and scalars.
 * @param fn Function called for `x`.
 * @param x... Scalar values.
 */
template<class T>
  requires (!is_simd_accessible<T>)
inline void elementwise(auto&& fn, T&& x, auto&&... y)
{
  fn(x, y...);
}

template<class T>
  requires (!is_simd_accessible<T>)
inline void elementwise_with_index(auto&& fn, T&& x, auto&&... y)
{
  fn(x, y...);
}

/**
 * Used in functors for `elementwise`, which write to the elements. The function and its overload is used
 * to distinguish between usual value types and simd::reference types.
 * @param x Element value.
 * @return Reference to `x`.
 */
inline auto& element_write(std::copyable auto& x)
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
inline std::remove_reference_t<T>&& element_write(T&& x)
{
  return static_cast<typename std::remove_reference<T>::type&&>(x) ;
}

/**
 * Returns a particular element of a simd type. This overload for non-simd types returns the passed value as
 * first element. For other elements the program is ill-formed.
 * @tparam I Number of the element. Must be 0.
 * @param x Value.
 * @return Constant reference to `x`.
 */
template<int I>
inline const auto& get_element(const auto& x)
{
  static_assert(I == 0);
  return x;
}

/**
 * Returns a particular element of a simd type.
 * @tparam I Number of the element. Must be smaller then `x.size()`.
 * @param x Simd value.
 * @return Value of `x[I]`.
 */
template<int I>
inline decltype(auto) get_element(const is_simd auto& x)
{
  static_assert(I < x.size());
  return x[I];
}

} //namespace simd_access

#endif //SIMD_ACCESS_ELEMENT_ACCESS
