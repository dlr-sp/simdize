// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Class for accessing simd values via simd indices.
 */

#ifndef SIMD_ACCESS_VALUE_ACCESS
#define SIMD_ACCESS_VALUE_ACCESS

#include "simd_access/operator_overload.hpp"
#include "simd_access/load_store.hpp"

namespace simd_access
{

/// Creates a binary operator overload for a simd value access as member.
/**
 * @param op Token for a binary operator (e.g. +,-,*,/).
 */
#define VALUE_ACCESS_BIN_OP( op ) \
  auto operator op(const auto& source) { return to_simd() op source; }

/// Creates a binary assignment operator overload for a simd value access.
/**
 * @param op Token for a binary operator (e.g. +,-,*,/).
 */
#define VALUE_ACCESS_BIN_ASSIGNMENT_OP( op ) \
  void operator op##=(const auto& source) && { store<ElementSize>(location_, to_simd() op source); }

/// Creates a binary assignment and a binary operator overload for a simd value access.
/**
 * @param op Token for a binary operator (e.g. +,-,*,/).
 */
#define VALUE_ACCESS_MEMBER_OPS( op ) \
  VALUE_ACCESS_BIN_OP( op ) \
  VALUE_ACCESS_BIN_ASSIGNMENT_OP( op )

/// Creates a global binary operator overload for a simd value access.
/**
 * @param op Token for a binary operator (e.g. +,-,*,/).
 */
#define VALUE_ACCESS_SCALAR_BIN_OP( op ) \
  template<class Location, size_t ElementSize> \
  inline auto operator op(const auto& o1, const value_access<Location, ElementSize>& o2) \
  { \
    return o1 op o2.to_simd(); \
  }

/// Class representing a simd-access (read or write) to a memory location.
/**
 * @tparam ElementSize Size of the array elements, which (or one of its members) are accessed by the simd index.
 * @tparam Location Type of the location of the simd data.
 */
template<class Location, size_t ElementSize>
class value_access :
  public member_overload<typename Location::value_type, value_access<Location, ElementSize>>,
  public cast_overload<typename Location::value_type, value_access<Location, ElementSize>>
{
public:
  /// Assignment operator.
  /** Writes a simd value to the simdized memory location represented by this.
   * Implicit conversion is not supported, it should be done explicit by the user.
   * Assign chains are not supported (i.e. the operator returns nothing).
   * @param source Simd value, whose content is written.
   */
  void operator=(const auto& source) &&
  {
    store<ElementSize>(location_, source);
  }

  VALUE_ACCESS_MEMBER_OPS(+)
  VALUE_ACCESS_MEMBER_OPS(-)
  VALUE_ACCESS_MEMBER_OPS(*)
  VALUE_ACCESS_MEMBER_OPS(/)

  /// Transforms this to a simd value.
  /**
   * @return The simd-ized access represented by this transformed to a simd value.
   */
  auto to_simd() const
  {
    return load<ElementSize>(location_);
  }

  /// Experimental implementation of overloaded member operator, i.e. operator.()
  /**
   * Since `T` might be of non-class type, one cannot specify `auto T::*Member` as a template argument here.
   * @tparam Member Pointer to the member of the class T.
   * @return A `value_access` with a base address pointing to the accessed member of `base_`.
   */
  template<auto Member>
  auto dot_overload()
  {
    return make_value_access<ElementSize>(location_.template member_access<Member>());
  }

  /// Implementation of subscript operator for accesses to sub-array elements of `Location::value_type`.
  /**
   * @param i An index usable as index type for sub-array elements of `Location::value_type`.
   * @return A `value_access` with a base address pointing to the accessed array element of `base_`.
   */
  auto operator[](auto i) const
  {
    return make_value_access<ElementSize>(location_.array_access(i));
  }

  /// Constructor.
  /**
   * @param location The location specification of the accessed simd variable.
   */
  value_access(const Location& location) :
    location_(location)
  {}

private:
  /// The location specification of the accessed simd variable.
  Location location_;
};

/// Factory function for `value_access`.
/**
 * @tparam ElementSize Size of the array elements, which (or one of its members) are accessed by the simd index.
 * @tparam Location Deduced type of the location of the simd data.
 * @param location The location specification of the accessed simd variable.
 * @return A `value_access` object representing a simd-access (read or write) to a memory location.
 */
template<size_t ElementSize, class Location>
inline auto make_value_access(const Location& location)
{
  return value_access<Location, ElementSize>(location);
}

VALUE_ACCESS_SCALAR_BIN_OP(+)
VALUE_ACCESS_SCALAR_BIN_OP(-)
VALUE_ACCESS_SCALAR_BIN_OP(*)
VALUE_ACCESS_SCALAR_BIN_OP(/)

template<class PotentialSimdType>
concept has_to_simd =
  requires(PotentialSimdType x) { x.to_simd(); };

} //namespace simd_access

#endif //SIMD_ACCESS_VALUE_ACCESS
