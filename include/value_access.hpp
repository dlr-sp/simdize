// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Class for accessing simd values via simd indices.
 * This file demonstrates a possible approach to overload the member access operator.
 */

#ifndef SIMD_ACCESS_VALUE_ACCESS
#define SIMD_ACCESS_VALUE_ACCESS

#include "member_overload.hpp"
#include "load_store.hpp"

namespace simd_access
{

#define VALUE_ACCESS_BIN_OP( op ) \
  template<class U> auto operator op(const U& source) { return to_simd() op source; }

#define VALUE_ACCESS_BIN_ASSIGNMENT_OP( op ) \
  template<class U> void operator op##=(const U& source) && { store<ElementSize>(base_, index_, to_simd() op source); }

#define VALUE_ACCESS_MEMBER_OPS( op ) \
  VALUE_ACCESS_BIN_OP( op ) \
  VALUE_ACCESS_BIN_ASSIGNMENT_OP( op )

#define VALUE_ACCESS_SCALAR_BIN_OP( op ) \
  template<class T, class Index, size_t ElementSize> \
  auto operator op(const std::type_identity_t<T>& o1, const value_access<T, Index, ElementSize>& o2) \
  { \
    return o1 op o2.to_simd(); \
  }

/// Class representing a simd-access (read or write) to a memory location.
/**
 * @tparam T Type of the value, that (or one of its members) is accessed.
 * @tparam Index Type of the simd index. May be `simd_index` or `simd_index_array`.
 * @tparam ElementSize Size of the array elements, which (or one of its members) are accessed by the simd index.
 */
template<class T, class Index, size_t ElementSize>
class value_access : public member_overload<T, value_access<T, Index, ElementSize>>
{
public:
  /// Assignment operator.
  /** Writes a simd value to the simdized memory location represented by this.
   * Implicit conversion is not supported, it should be done explicit by the user.
   * Assign chains are not supported (i.e. the operator returns nothing).
   * @param source Simd value, whose content is written.
   */
  void operator=(const stdx::fixed_size_simd<T, Index::size()>& source) &&
  {
    store<ElementSize>(base_, index_, source);
  }

  VALUE_ACCESS_MEMBER_OPS(+)
  VALUE_ACCESS_MEMBER_OPS(-)
  VALUE_ACCESS_MEMBER_OPS(*)
  VALUE_ACCESS_MEMBER_OPS(/)

  /// Cast operator to a simd value.
  /** Transforms this to a simd value.
   * @return The simd-ized access represented by this transformed to a simd value.
   */
  operator stdx::fixed_size_simd<std::remove_const_t<T>, Index::size()>() const
  {
    return to_simd();
  }

  /// Transforms this to a simd value.
  /**
   * @return The simd-ized access represented by this transformed to a simd value.
   */
  auto to_simd() const
  {
    return load<ElementSize>(base_, index_);
  }

  /// Implementation of overloaded member operator, i.e. operator.()
  /**
   * Since `T` might be of non-class type, one cannot specify `auto T::*Member` as a template argument here.
   * @tparam Member Pointer to the member of the class T.
   * @return A `value_access` with a base address pointing to the accessed member of `base_`.
   */
  template<auto Member>
  auto dot_overload()
  {
    using ResultType = value_access<std::remove_reference_t<decltype(base_->*Member)>, Index, ElementSize>;
    return ResultType(&(base_->*Member), index_);
  }

  /// Implementation of subscript operator for accesses to sub-array elements, if `T` is an array type.
  /**
   * @return A `value_access` with a base address pointing to the accessed array element of `base_`.
   */
  auto operator[](int i)
  {
    using ResultType = value_access<std::remove_pointer_t<decltype((*base_) + i)>, Index, ElementSize>;
    return ResultType((*base_) + i, index_);
  }

  /// Constructor.
  value_access(T* base, const Index& index) :
    base_(base),
    index_(index)
  {}

private:
  T* base_;
  const Index& index_;
};

VALUE_ACCESS_SCALAR_BIN_OP(+)
VALUE_ACCESS_SCALAR_BIN_OP(-)
VALUE_ACCESS_SCALAR_BIN_OP(*)
VALUE_ACCESS_SCALAR_BIN_OP(/)

} //namespace simd_access

#endif //SIMD_ACCESS_VALUE_ACCESS
