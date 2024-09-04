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

template<typename... Args>
struct vector : std::vector<Args...>
{
  using std::vector<Args...>::vector;

  auto operator[](const is_index auto& index)
  {
    return LValueSeparator<true>::to_simd(*this, index);
  }

  auto operator[](const is_index auto& index) const
  {
    return LValueSeparator<true>::to_simd(*this, index);
  }

  using std::vector<Args...>::operator[];
};

} //namespace simd_access

#endif //SIMD_ACCESS_VECTOR
