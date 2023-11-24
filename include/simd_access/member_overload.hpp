// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Base class for member operator overload
 * This file demonstrates a possible approach to overload the member access operator.
 */

#ifndef SIMD_ACCESS_MEMBER_OVERLOAD
#define SIMD_ACCESS_MEMBER_OVERLOAD

#include <type_traits>

namespace simd_access
{

/// class concept: if fulfilled, T is syntactically allowed to have members.
template<class T> concept is_class = std::is_class_v<T>;
template<class T> concept is_not_class = !std::is_class_v<T>;

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
template<is_not_class T, typename Derived>
struct member_overload<T, Derived> {};

/// Specialization for class types.
template<is_class T, typename Derived>
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

} //namespace simd_access

#endif //SIMD_ACCESS_MEMBER_OVERLOAD
