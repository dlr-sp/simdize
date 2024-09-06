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

template<typename Base>
struct SimdIndexOperator : Base
{
  using Base::Base;

  auto operator[](const is_index auto& index)
  {
    return LValueSeparator<true>::to_simd(*this, index);
  }

  auto operator[](const is_index auto& index) const
  {
    return LValueSeparator<true>::to_simd(*this, index);
  }

  using Base::operator[];
};

template<typename... Args>
using vector = SimdIndexOperator<std::vector<Args...>>;

} //namespace simd_access

#endif //SIMD_ACCESS_VECTOR
