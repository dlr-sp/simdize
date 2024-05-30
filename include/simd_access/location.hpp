// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining locations of simdized data.
 */

#ifndef SIMD_ACCESS_LOCATION
#define SIMD_ACCESS_LOCATION

#include <type_traits>

namespace simd_access
{

template<class T, int SimdSize>
struct linear_location
{
  using value_type = T;
  T* base_;

  template<auto Member>
  auto member_access() const
  {
    return linear_location<std::remove_reference_t<decltype(std::declval<T>().*Member)>, SimdSize>{&(base_->*Member)};
  }

  auto array_access(auto i) const
  {
    return linear_location<std::remove_reference_t<decltype((*base_)[i])>, SimdSize>{&((*base_)[i])};
  }
};

template<class T, int SimdSize, class ArrayType>
struct indexed_location
{
  using value_type = T;
  T* base_;
  const ArrayType& indices_;

  template<auto Member>
  auto member_access() const
  {
    return indexed_location<std::remove_reference_t<decltype(std::declval<T>().*Member)>, SimdSize, ArrayType>
      {&(base_->*Member), indices_};
  }

  auto array_access(auto i) const
  {
    return indexed_location<std::remove_reference_t<decltype((*base_)[i])>, SimdSize, ArrayType>
      {&((*base_)[i]), indices_};
  }
};

template<class T, int SimdSize>
struct random_location
{
  using value_type = T;
  T* base_[SimdSize];

  template<auto Member>
  auto member_access() const
  {
    random_location<std::remove_reference_t<decltype(std::declval<T>().*Member)>, SimdSize> result;
    for (int i = 0; i < SimdSize; ++i)
    {
      result.base_[i] = &(base_[i]->*Member);
    }
  }

  auto array_access(auto i) const
  {
    random_location<std::remove_pointer_t<decltype(*std::declval<T>())>, SimdSize> result;
    for (int k = 0; k < SimdSize; ++k)
    {
      result.base_[k] = &((*base_[k])[i]);
    }
  }
};

} //namespace simd_access

#endif //SIMD_ACCESS_LOCATION



