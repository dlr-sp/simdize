// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Base class for operator overloads
 * This file demonstrates a possible approach to overload the member access operator.
 */

#ifndef SIMD_ACCESS_OPERATOR_OVERLOAD
#define SIMD_ACCESS_OPERATOR_OVERLOAD

#include <type_traits>
#include "base.hpp"

namespace simd_access
{

/// Base class for overloaded member operator.
/**
 * This CRTP base class is needed, if the member operator is overloaded and the types, whose members are potentially
 * accessed might not be classes.
 * @tparam T Type of the class, whose members are accessed.
 * @tparam Derived subclass of this class. It provides a member function `dot_overload` using a less
 * restrictive template syntax.
 */
template<typename T, typename Derived>
struct member_overload;

/// Empty specialization for non-class types. Avoids syntax errors on the template parameter for any non-class T.
template<typename T, typename Derived> requires (!std::is_class_v<T>)
struct member_overload<T, Derived> {};

/// Specialization for class types.
template<typename T, typename Derived> requires (std::is_class_v<T>)
struct member_overload<T, Derived>
{
  /// Overloaded member operator, i.e. operator.()
  /**
   * An overload of `operator.()` for data members must have exactly the form below. The type of the object, whose
   * member is accessed, must be specified in the template argument to enable the compiler to look up for
   * the appropriate member in the expression using the overloaded `operator.()`.
   * @tparam Member Pointer to the member of the class T.
   * @return The value returned by `Derived::dot_overload`.
   */
  template<auto T::*Member>
  auto dot()
  {
    return (static_cast<Derived*>(this))->template dot_overload<Member>();
  }
};

/// Base class for overloaded type cast operator.
/**
 * This CRTP base class restricts the availability of the cast `operator auto()` to value types, which can be simd
 * value types. This prevents instantiation errors (gcc #115216).
 * @tparam T Type of scalar value.
 * @tparam Derived subclass of this class. It provides a member function `to_simd`.
 */
template<typename T, typename Derived>
struct cast_overload;

/// Empty specialization for non-simd_arithmetic types. There is no implicit cast to a simd type.
template<typename T, typename Derived> requires(!simd_arithmetic<T>)
struct cast_overload<T, Derived> {};

/// Specialization for simd_arithmetic types.
template<simd_arithmetic T, typename Derived>
struct cast_overload<T, Derived>
{
  /// Cast operator to a simd value.
  /** Transforms this to a simd value.
   * @return The simd-ized access represented by this transformed to a simd value.
   */
  operator auto() const
  {
    return (static_cast<const Derived*>(this))->to_simd();
  }
};

} //namespace simd_access

#endif //SIMD_ACCESS_OPERATOR_OVERLOAD
