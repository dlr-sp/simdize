// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes representing simd indices to either a consecutive sequence of elements or indirect indexed elements.
 */

#ifndef SIMD_ACCESS_INDEX
#define SIMD_ACCESS_INDEX

#include <concepts>
#include <type_traits>

#include "simd_access/base.hpp"
#include "simd_access/location.hpp"

namespace simd_access
{

template<class Location, size_t ElementSize>
class value_access;

/// Class representing a simd index to a consecutive sequence of elements.
/**
 * @tparam SimdSize Length of the simd sequence.
 * @tparam IndexType Type of the scalar index.
 */
template<int SimdSize, class IndexType = size_t>
struct index
{
  /// Return the length of the simd sequence.
  /**
   * @return The length of the simd sequence.
   */
  static constexpr int size() { return SimdSize; }

  /// Return the scalar index of a vector lane.
  /**
   * @param i Index in the vector must be in the range [0, SimdSize) .
   * @return The scalar index at vector lane i, i.e. index_ + i.
   */
  auto scalar_index(int i) const { return index_ + IndexType(i); }

  /// The index, at which the sequence starts.
  IndexType index_;

  /// A reverse overloaded operator[] for simdized array accesses, since global operator[] is not allowed (yet).
  /**
   * @tparam T Data type of the elements in the array.
   * @param data Pointer to the array.
   * @return A value_access representing a simd access expression to a consecutive sequence of elements in an array.
   */
  template<class T>
  auto operator[](T* data) const
  {
    return value_access<linear_location<T, SimdSize>, sizeof(T)>(linear_location<T, SimdSize>{data + index_});
  }

  /// Transforms this to a simd value.
  /**
   * @return The value represented by this transformed to a simd value.
   */
  auto to_simd() const
  {
    return stdx::fixed_size_simd<IndexType, SimdSize>([this](auto i){ return IndexType(index_ + i); });
  }
};

template<class PotentialIndexType>
concept simd_index =
  (stdx_simd<PotentialIndexType> && std::is_integral_v<typename PotentialIndexType::value_type>) ||
  requires(std::remove_cvref_t<PotentialIndexType> x) { []<int SimdSize, class IndexType>(index<SimdSize, IndexType>&){}(x); };

/// TODO: Introduce masked_index to support e.g. residual masked loops.

/// Returns the scalar index of a specific vector lane for a linear index.
/**
 * @tparam SimdSize Deduced simd size.
 * @tparam IndexType Deduced type of the scalar index.
 * @param idx Linear simd index.
 * @param i Vector lane.
 * @return The scalar index at vector lane `i`, i.e. `idx.start + i`.
 */
template<int SimdSize, class IndexType>
inline auto scalar_index(const index<SimdSize, IndexType>& idx, auto i)
{
  return idx.scalar_index(i);
}

/// Returns the scalar index of a specific vector lane for an indirect index u.
/**
 * @tparam IndexType Deduced integral type of the scalar index.
 * @tparam Abi Deduced abi of the `simd` paramter.
 * @param idx Indirect simd index.
 * @param i Vector lane.
 * @return The scalar index at vector lane `i`, i.e. `idx[i]`.
 */
template<std::integral IndexType, class Abi>
inline auto scalar_index(const stdx::simd<IndexType, Abi>& idx, auto i)
{
  return idx[i];
}

/// Returns true, if the argument is a simd index (i.e. fullfills the concept `simd_index`).
/**
 * @tparam IndexType Deduced integral type of the scalar index.
 * @tparam Abi Deduced abi of the `simd` paramter.
 * @param idx Indirect simd index.
 * @param i Vector lane.
 * @return The scalar index at vector lane `i`, i.e. `idx[i]`.
 */
 constexpr inline auto is_simd_index(auto&& idx)
{
  return simd_index<decltype(idx)>;
}

} //namespace simd_access

#endif //SIMD_ACCESS_INDEX
