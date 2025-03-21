// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief An extended std::vector.
 */

#ifndef SIMD_ACCESS_VECTOR
#define SIMD_ACCESS_VECTOR

#include <vector>

#include "simd_access/base.hpp"
#include "simd_access/index.hpp"
#include "simd_access/simd_access.hpp"

namespace simd_access
{

/// Helper class, which injects an overloaded operator[] for simd indices to a given class.
/**
 * @tparam Base Class, into which the operator[] is injected.
 */
template<typename Base>
struct index_operator : Base
{
  /// Make constructors of the base class accessible.
  using Base::Base;

  /// Overloaded operator[] for simd indices (non-const version).
  /**
   * @param index Simd index.
   * @return A value access object (see \ref value_access) for simd accesses, which can be used as lhs in assignments.
   */
  auto operator[](const simd_index auto& index)
  {
    return LValueSeparator<true>::to_simd(*this, index);
  }

  /// Overloaded operator[] for simd indices (const version).
  /**
   * @param index Simd index.
   * @return A value access object (see \ref value_access) for simd accesses.
   */
  auto operator[](const simd_index auto& index) const
  {
    return LValueSeparator<true>::to_simd(*this, index);
  }

  /// Make operator[] of the base class accessible.
  using Base::operator[];
};

/// A shortcut type for a std::vector with an overloaded operator[] for simd indices.
/**
 * @tparam Args Template Arguments passed to std::vector.
 */
template<typename... Args>
using vector = index_operator<std::vector<Args...>>;

} //namespace simd_access

#endif //SIMD_ACCESS_VECTOR
