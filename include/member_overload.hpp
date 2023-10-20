// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Base class for member operator overload
 * This file demonstrates a possible approach to overload the member access operator.
 */

#include <type_traits>

/// class concept: if fulfilled, T is syntactically allowed to have members.
template<class T> concept is_class = std::is_class_v<T>;
template<class T> concept is_not_class = !std::is_class_v<T>;

/// Base class for overloaded member operator.
/**
 * This CRTP base class is needed, if the member operator is overloaded and the types, whose members are potentially
 * accessed might not be classes.
 * @tparam T Type of the class, whose members are accessed.
 * @tparam Derived Subclass of this class. It provides a member function dot_overload using a less
 * restrictive template syntax.
 */
template<typename T, typename Derived>
struct member_overload;

/// Empty specialization for non-class types. Avoids syntax errors on the template parameter any for non-class T.
template<is_not_class T, typename Derived>
struct member_overload<T, Derived> {};

/// Specialization for class types.
template<is_class T, typename Derived>
struct member_overload<T, Derived>
{
  /// Overloaded member operator, i.e. operator.()
  /**
   * An overload of operator.() for data members must have exactly the form below. The type of the object, of which
   * a member is accessed must be specified in the template argument list to enable the compiler to look up for
   * the appropriate member.
   * @tparam Member Pointer to the member of the class T.
   */
  template<auto T::*Member>
  auto dot()
  {
    return (static_cast<Derived*>(this))->template dot_overload<Member>();
  }
};
