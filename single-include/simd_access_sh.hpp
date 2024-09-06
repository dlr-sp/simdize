// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Helper functions and a macro to mark a simd access.
 */

#ifndef SIMD_ACCESS_MAIN
#define SIMD_ACCESS_MAIN

#include <concepts>

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <array>
#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

template<class T, int SimdSize>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

template<class PotentialSimdType>
concept is_stdx_simd =
  requires(PotentialSimdType x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept is_simd =
  is_stdx_simd<PotentialSimdType> ||
  requires(PotentialSimdType x) { []<class T, int SimdSize>(universal_simd<T, SimdSize>&){}(x); };

template<class T, int SimdSize>
struct auto_simd
{
  using type = universal_simd<T, SimdSize>;
};

template<simd_arithmetic T, int SimdSize>
struct auto_simd<T, SimdSize>
{
  using type = stdx::fixed_size_simd<T, SimdSize>;
};

template<class T, int SimdSize>
using auto_simd_t = typename auto_simd<T, SimdSize>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Elementwise access of simd elements.
 */

#ifndef SIMD_ACCESS_ELEMENT_ACCESS
#define SIMD_ACCESS_ELEMENT_ACCESS

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <array>
#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

template<class T, int SimdSize>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

template<class PotentialSimdType>
concept is_stdx_simd =
  requires(PotentialSimdType x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept is_simd =
  is_stdx_simd<PotentialSimdType> ||
  requires(PotentialSimdType x) { []<class T, int SimdSize>(universal_simd<T, SimdSize>&){}(x); };

template<class T, int SimdSize>
struct auto_simd
{
  using type = universal_simd<T, SimdSize>;
};

template<simd_arithmetic T, int SimdSize>
struct auto_simd<T, SimdSize>
{
  using type = stdx::fixed_size_simd<T, SimdSize>;
};

template<class T, int SimdSize>
using auto_simd_t = typename auto_simd<T, SimdSize>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

namespace simd_access
{

/**
 * Call a function for each scalar of a simd value.
 * @param x Simd value.
 * @param fn Function called for each scalar value of `x`. If the function intends to write to the element, then it
 *   must forward its argument as in `[&](auto&& y) { element_write(y) = ...; }`.
 */
inline void elementwise(is_simd auto&& x, auto&& fn)
{
  for (int i = 0; i < x.size(); ++i)
  {
    fn(x[i]);
  }
}

/**
 * Call a function for the scalar value `x`.
 * This overload can be used with a generic functor to uniformly access scalar values of simds and scalars.
 * @param x Scalar value.
 * @param fn Function called for `x`.
 */
inline void elementwise(auto&& x, auto&& fn)
{
  fn(x);
}

/**
 * Used in functors for `elementwise`, which write to the elements. The function and its overload is used
 * to distinguish between usual value types and simd::reference types.
 * @param x Element value.
 * @return Reference to `x`.
 */
inline auto& element_write(std::copyable auto& x)
{
  return x;
}

/**
 * Overloaded function intended to be called for simd::reference types, which are not copyable.
 * @param T Type of the element value.
 * @param x Element value.
 * @return Rvalue reference to `x`, which can be used in assignment (since only
 *   `simd::reference::operator=(auto&&) &&` is available.
 */
template<class T>
inline std::remove_reference_t<T>&& element_write(T&& x)
{
  return static_cast<typename std::remove_reference<T>::type&&>(x) ;
}

/**
 * Returns a particular element of a simd type. This overload for non-simd types returns the passed value as
 * first element. For other elements the program is ill-formed.
 * @tparam I Number of the element. Must be 0.
 * @param x Value.
 * @return Constant reference to `x`.
 */
template<int I>
inline const auto& get_element(const auto& x)
{
  static_assert(I == 0);
  return x;
}

/**
 * Returns a particular element of a simd type.
 * @tparam I Number of the element. Must be smaller then `x.size()`.
 * @param x Simd value.
 * @return Value of `x[I]`.
 */
template<int I>
inline decltype(auto) get_element(const is_simd auto& x)
{
  static_assert(I < x.size());
  return x[I];
}

} //namespace simd_access

#endif //SIMD_ACCESS_ELEMENT_ACCESS

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes representing simd indices to either a consecutive sequence of elements or indirect indexed elements.
 */

#ifndef SIMD_ACCESS_INDEX
#define SIMD_ACCESS_INDEX

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <array>
#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

template<class T, int SimdSize>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

template<class PotentialSimdType>
concept is_stdx_simd =
  requires(PotentialSimdType x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept is_simd =
  is_stdx_simd<PotentialSimdType> ||
  requires(PotentialSimdType x) { []<class T, int SimdSize>(universal_simd<T, SimdSize>&){}(x); };

template<class T, int SimdSize>
struct auto_simd
{
  using type = universal_simd<T, SimdSize>;
};

template<simd_arithmetic T, int SimdSize>
struct auto_simd<T, SimdSize>
{
  using type = stdx::fixed_size_simd<T, SimdSize>;
};

template<class T, int SimdSize>
using auto_simd_t = typename auto_simd<T, SimdSize>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

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

#include <type_traits>

namespace simd_access
{

template<class Location, size_t ElementSize>
class value_access;

/// Class representing a simd index to a consecutive sequence of elements.
/**
 * @tparam SimdSize Length of the simd sequence.
 * @tparam IndexType Type of the index.
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

/// Class representing a simd index to indirect indexed elements in an array.
/**
 * @tparam SimdSize Length of the simd sequence.
 * @tparam ArrayType Type of the array, which stores the indices. Defaults to std::array<size_t, SimdSize>, but could
 *   be stdx::fixed_size_simd<size_t, SimdSize>. Also size_t can be exchanged for another integral type.
 *   It is also possible to specify e.g. `int*` and thus use a pointer to a location in a larger array.
 */
template<int SimdSize, class ArrayType = std::array<size_t, SimdSize>>
struct index_array
{
  /// Return the length of the simd sequence.
  /**
   * @return The length of the simd sequence.
   */
  static constexpr int size() { return SimdSize; }

  /// Return the scalar index of a vector lane.
  /**
   * @param i Index in the vector must be in the range [0, SimdSize) .
   * @return The scalar index at vector lane i, i.e. index_[i].
   */
  auto scalar_index(int i) const { return index_[i]; }

  /// The index array, the n'th entry defines the index of the n'th element in the simd type.
  ArrayType index_;

  /// A reverse overloaded operator[] for simdized array accesses, since global operator[] is not allowed (yet).
  /**
   * @tparam T Data type of the elements in the array.
   * @param data Pointer to the array.
   * @return A value_access representing a simd access expression to indirect indexed elements in an array.
   */
  template<class T>
  auto operator[](T* data) const
  {
    return value_access<indexed_location<T, SimdSize, ArrayType>, sizeof(T)>
      (indexed_location<T, SimdSize, ArrayType>{data, index_});
  }
};

template<class PotentialIndexType>
concept is_index =
  (is_stdx_simd<PotentialIndexType> && std::is_integral_v<typename PotentialIndexType::value_type>) ||
  requires(PotentialIndexType x) { []<int SimdSize, class IndexType>(index<SimdSize, IndexType>&){}(x); } ||
  requires(PotentialIndexType x) { []<int SimdSize, class ArrayType>(index_array<SimdSize, ArrayType>&){}(x); };

/// TODO: Introduce masked_index and masked_index_array to support e.g. residual masked loops.

template<int SimdSize, class IndexType>
inline auto get_index(const index<SimdSize, IndexType>& idx, auto i)
{
  return idx.index_ + i;
}

template<int SimdSize, class ArrayType>
inline auto get_index(const index_array<SimdSize, ArrayType>& idx, auto i)
{
  return idx.scalar_index(i);
}

template<class IndexType, class Abi>
inline auto get_index(const stdx::simd<IndexType, Abi>& idx, auto i)
{
  return idx[i];
}

} //namespace simd_access

#endif //SIMD_ACCESS_INDEX

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Global functions to load and store simd values to using several addressing modes.
 */

#ifndef SIMD_LOAD_STORE
#define SIMD_LOAD_STORE

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <array>
#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

template<class T, int SimdSize>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

template<class PotentialSimdType>
concept is_stdx_simd =
  requires(PotentialSimdType x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept is_simd =
  is_stdx_simd<PotentialSimdType> ||
  requires(PotentialSimdType x) { []<class T, int SimdSize>(universal_simd<T, SimdSize>&){}(x); };

template<class T, int SimdSize>
struct auto_simd
{
  using type = universal_simd<T, SimdSize>;
};

template<simd_arithmetic T, int SimdSize>
struct auto_simd<T, SimdSize>
{
  using type = stdx::fixed_size_simd<T, SimdSize>;
};

template<class T, int SimdSize>
using auto_simd_t = typename auto_simd<T, SimdSize>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

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

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes representing simd indices to either a consecutive sequence of elements or indirect indexed elements.
 */

#ifndef SIMD_ACCESS_INDEX
#define SIMD_ACCESS_INDEX

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <array>
#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

template<class T, int SimdSize>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

template<class PotentialSimdType>
concept is_stdx_simd =
  requires(PotentialSimdType x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept is_simd =
  is_stdx_simd<PotentialSimdType> ||
  requires(PotentialSimdType x) { []<class T, int SimdSize>(universal_simd<T, SimdSize>&){}(x); };

template<class T, int SimdSize>
struct auto_simd
{
  using type = universal_simd<T, SimdSize>;
};

template<simd_arithmetic T, int SimdSize>
struct auto_simd<T, SimdSize>
{
  using type = stdx::fixed_size_simd<T, SimdSize>;
};

template<class T, int SimdSize>
using auto_simd_t = typename auto_simd<T, SimdSize>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

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

#include <type_traits>

namespace simd_access
{

template<class Location, size_t ElementSize>
class value_access;

/// Class representing a simd index to a consecutive sequence of elements.
/**
 * @tparam SimdSize Length of the simd sequence.
 * @tparam IndexType Type of the index.
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

/// Class representing a simd index to indirect indexed elements in an array.
/**
 * @tparam SimdSize Length of the simd sequence.
 * @tparam ArrayType Type of the array, which stores the indices. Defaults to std::array<size_t, SimdSize>, but could
 *   be stdx::fixed_size_simd<size_t, SimdSize>. Also size_t can be exchanged for another integral type.
 *   It is also possible to specify e.g. `int*` and thus use a pointer to a location in a larger array.
 */
template<int SimdSize, class ArrayType = std::array<size_t, SimdSize>>
struct index_array
{
  /// Return the length of the simd sequence.
  /**
   * @return The length of the simd sequence.
   */
  static constexpr int size() { return SimdSize; }

  /// Return the scalar index of a vector lane.
  /**
   * @param i Index in the vector must be in the range [0, SimdSize) .
   * @return The scalar index at vector lane i, i.e. index_[i].
   */
  auto scalar_index(int i) const { return index_[i]; }

  /// The index array, the n'th entry defines the index of the n'th element in the simd type.
  ArrayType index_;

  /// A reverse overloaded operator[] for simdized array accesses, since global operator[] is not allowed (yet).
  /**
   * @tparam T Data type of the elements in the array.
   * @param data Pointer to the array.
   * @return A value_access representing a simd access expression to indirect indexed elements in an array.
   */
  template<class T>
  auto operator[](T* data) const
  {
    return value_access<indexed_location<T, SimdSize, ArrayType>, sizeof(T)>
      (indexed_location<T, SimdSize, ArrayType>{data, index_});
  }
};

template<class PotentialIndexType>
concept is_index =
  (is_stdx_simd<PotentialIndexType> && std::is_integral_v<typename PotentialIndexType::value_type>) ||
  requires(PotentialIndexType x) { []<int SimdSize, class IndexType>(index<SimdSize, IndexType>&){}(x); } ||
  requires(PotentialIndexType x) { []<int SimdSize, class ArrayType>(index_array<SimdSize, ArrayType>&){}(x); };

/// TODO: Introduce masked_index and masked_index_array to support e.g. residual masked loops.

template<int SimdSize, class IndexType>
inline auto get_index(const index<SimdSize, IndexType>& idx, auto i)
{
  return idx.index_ + i;
}

template<int SimdSize, class ArrayType>
inline auto get_index(const index_array<SimdSize, ArrayType>& idx, auto i)
{
  return idx.scalar_index(i);
}

template<class IndexType, class Abi>
inline auto get_index(const stdx::simd<IndexType, Abi>& idx, auto i)
{
  return idx[i];
}

} //namespace simd_access

#endif //SIMD_ACCESS_INDEX

namespace simd_access
{

/**
 * Stores a simd value to a memory location defined by a base address and an linear index. The simd elements are
 * stored at the positions base, base+ElementSize, base+2*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of a simd element.
 * @tparam SimdSize Deduced vector size of the simd type.
 * @param location Address of the memory location, at which the first simd element is stored.
 * @param source Simd value to be stored.
 */
template<size_t ElementSize, simd_arithmetic T, int SimdSize>
inline void store(const linear_location<T, SimdSize>& location, const stdx::fixed_size_simd<T, SimdSize>& source)
{
  if constexpr (sizeof(T) == ElementSize)
  {
    source.copy_to(location.base_, stdx::element_aligned);
  }
  else
  {
    // scatter with constant pitch
    for (int i = 0; i < SimdSize; ++i)
    {
      *reinterpret_cast<T*>(reinterpret_cast<char*>(location.base_) + ElementSize * i) = source[i];
    }
  }
}

/**
 * Stores a simd value to a memory location defined by a base address and an indirect index. The simd elements are
 * stored at the positions base+indices[0]*ElementSize, base+indices[1]*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of a simd element.
 * @tparam SimdSize Deduced vector size of the simd type.
 * @tparam ArrayType Deduced type of the array storing the indices.
 * @param location Address and indices of the memory location.
 * @param source Simd value to be stored.
 */
template<size_t ElementSize, simd_arithmetic T, int SimdSize, class ArrayType>
inline void store(const indexed_location<T, SimdSize, ArrayType>& location,
  const stdx::fixed_size_simd<T, SimdSize>& source)
{
  // scatter with indirect indices
  for (int i = 0; i < SimdSize; ++i)
  {
    *reinterpret_cast<T*>(reinterpret_cast<char*>(location.base_) + ElementSize * location.indices_[i]) = source[i];
  }
}

/**
 * Loads a simd value from a memory location defined by a base address and an linear index. The simd elements to be
 * loaded are located at the positions base, base+ElementSize, base+2*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Type of a simd element.
 * @tparam SimdSize Vector size of the simd type.
 * @param location Address of the memory location, at which the first scalar element is stored.
 * @return A simd value.
 */
template<size_t ElementSize, simd_arithmetic T, int SimdSize>
inline auto load(const linear_location<T, SimdSize>& location)
{
  using ResultType = stdx::fixed_size_simd<std::remove_const_t<T>, SimdSize>;
  if constexpr (sizeof(T) == ElementSize)
  {
    return ResultType(location.base_, stdx::element_aligned);
  }
  else
  {
    // gather with constant pitch
    return ResultType([&](int i)
      {
        return *reinterpret_cast<const T*>(reinterpret_cast<const char*>(location.base_) + ElementSize * i);
      });
  }
}

/**
 * Loads a simd value from a memory location defined by a base address and an indirect index. The simd elements to be
 * loaded are stored at the positions base+indices[0]*ElementSize, base+indices[1]*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of a simd element.
 * @tparam SimdSize Deduced vector size of the simd type.
 * @tparam ArrayType Deduced type of the array storing the indices.
 * @param location Address and indices of the memory location.
 * @return A simd value.
 */
template<size_t ElementSize, simd_arithmetic T, int SimdSize, class ArrayType>
inline auto load(const indexed_location<T, SimdSize, ArrayType>& location)
{
  // gather with indirect indices
  return stdx::fixed_size_simd<std::remove_const_t<T>, SimdSize>([&](int i)
    {
      return *reinterpret_cast<const T*>
        (reinterpret_cast<const char*>(location.base_) + ElementSize * location.indices_[i]);
    });
}

/**
 * Creates a simd value from rvalues returned by the operator[] applied to `base`.
 * @tparam BaseType Type of an simd element.
 * @param base Base object.
 * @param idx Index.
 * @return A simd value.
 */
template<simd_arithmetic BaseType>
inline auto load_rvalue(auto&& base, const auto& idx)
{
  return stdx::fixed_size_simd<BaseType, idx.size()>([&](auto i) { return base[get_index(idx, i)]; });
}

/**
 * Creates a simd value from rvalues returned by the functor `subobject` applied to the range of elements defined by
 * the simd index `idx`.
 * @tparam BaseType Type of an simd element.
 * @param base Base object.
 * @param idx Index.
 * @param subobject Functor returning the sub-object.
 * @return A simd value.
 */
template<simd_arithmetic BaseType>
inline auto load_rvalue(auto&& base, const auto& idx, auto&& subobject)
{
  return stdx::fixed_size_simd<BaseType, idx.size()>([&](auto i) { return subobject(base[get_index(idx, i)]); });
}

} //namespace simd_access

#endif //SIMD_LOAD_STORE

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Functions looping over a given function in simd-style in a linear or indirect fashion.
 */

#ifndef SIMD_ACCESS_LOOP
#define SIMD_ACCESS_LOOP

#include <concepts>
#include <type_traits>
// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes representing simd indices to either a consecutive sequence of elements or indirect indexed elements.
 */

#ifndef SIMD_ACCESS_INDEX
#define SIMD_ACCESS_INDEX

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <array>
#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

template<class T, int SimdSize>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

template<class PotentialSimdType>
concept is_stdx_simd =
  requires(PotentialSimdType x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept is_simd =
  is_stdx_simd<PotentialSimdType> ||
  requires(PotentialSimdType x) { []<class T, int SimdSize>(universal_simd<T, SimdSize>&){}(x); };

template<class T, int SimdSize>
struct auto_simd
{
  using type = universal_simd<T, SimdSize>;
};

template<simd_arithmetic T, int SimdSize>
struct auto_simd<T, SimdSize>
{
  using type = stdx::fixed_size_simd<T, SimdSize>;
};

template<class T, int SimdSize>
using auto_simd_t = typename auto_simd<T, SimdSize>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

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

#include <type_traits>

namespace simd_access
{

template<class Location, size_t ElementSize>
class value_access;

/// Class representing a simd index to a consecutive sequence of elements.
/**
 * @tparam SimdSize Length of the simd sequence.
 * @tparam IndexType Type of the index.
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

/// Class representing a simd index to indirect indexed elements in an array.
/**
 * @tparam SimdSize Length of the simd sequence.
 * @tparam ArrayType Type of the array, which stores the indices. Defaults to std::array<size_t, SimdSize>, but could
 *   be stdx::fixed_size_simd<size_t, SimdSize>. Also size_t can be exchanged for another integral type.
 *   It is also possible to specify e.g. `int*` and thus use a pointer to a location in a larger array.
 */
template<int SimdSize, class ArrayType = std::array<size_t, SimdSize>>
struct index_array
{
  /// Return the length of the simd sequence.
  /**
   * @return The length of the simd sequence.
   */
  static constexpr int size() { return SimdSize; }

  /// Return the scalar index of a vector lane.
  /**
   * @param i Index in the vector must be in the range [0, SimdSize) .
   * @return The scalar index at vector lane i, i.e. index_[i].
   */
  auto scalar_index(int i) const { return index_[i]; }

  /// The index array, the n'th entry defines the index of the n'th element in the simd type.
  ArrayType index_;

  /// A reverse overloaded operator[] for simdized array accesses, since global operator[] is not allowed (yet).
  /**
   * @tparam T Data type of the elements in the array.
   * @param data Pointer to the array.
   * @return A value_access representing a simd access expression to indirect indexed elements in an array.
   */
  template<class T>
  auto operator[](T* data) const
  {
    return value_access<indexed_location<T, SimdSize, ArrayType>, sizeof(T)>
      (indexed_location<T, SimdSize, ArrayType>{data, index_});
  }
};

template<class PotentialIndexType>
concept is_index =
  (is_stdx_simd<PotentialIndexType> && std::is_integral_v<typename PotentialIndexType::value_type>) ||
  requires(PotentialIndexType x) { []<int SimdSize, class IndexType>(index<SimdSize, IndexType>&){}(x); } ||
  requires(PotentialIndexType x) { []<int SimdSize, class ArrayType>(index_array<SimdSize, ArrayType>&){}(x); };

/// TODO: Introduce masked_index and masked_index_array to support e.g. residual masked loops.

template<int SimdSize, class IndexType>
inline auto get_index(const index<SimdSize, IndexType>& idx, auto i)
{
  return idx.index_ + i;
}

template<int SimdSize, class ArrayType>
inline auto get_index(const index_array<SimdSize, ArrayType>& idx, auto i)
{
  return idx.scalar_index(i);
}

template<class IndexType, class Abi>
inline auto get_index(const stdx::simd<IndexType, Abi>& idx, auto i)
{
  return idx[i];
}

} //namespace simd_access

#endif //SIMD_ACCESS_INDEX

namespace simd_access
{

using ScalarResidualLoopT = std::integral_constant<int, 0>;
using VectorResidualLoopT = std::integral_constant<int, 1>;
constexpr auto ScalarResidualLoop = ScalarResidualLoopT();
constexpr auto VectorResidualLoop = VectorResidualLoopT();

/**
 * Linear simd-ized iteration over a function. The function is first called with a simd index and the remainder
 * loop is called with an integral index.
 * @tparam SimdSize Vector size.
 * @param start Start of the iteration range [start, end).
 * @param end End of the iteration range [start, end).
 * @param fn Generic function to be called. Takes one argument, whose type is either `index<SimdSize, IntegralType>`
 *   or `IntegralType`.
 * @param residualLoopPolicy Determines the execution policy of residual iterations. If `ScalarResidualLoop`, residual
 *   iterations are executed one by one. If `VectorResidualLoop`, residual iterations are executed vectorized. In that
 *   case the user is responsible for the handling of indices possbily extending the valid iteration range. Defaults
 *   to `ScalarResidualLoop`.
 */
template<int SimdSize, auto ... Args, typename ResidualLoopPolicyType = ScalarResidualLoopT>
inline void loop(std::integral auto start, std::integral auto end, auto&& fn,
  ResidualLoopPolicyType residualLoopPolicy = ScalarResidualLoop)
{
  using IndexType = std::common_type_t<decltype(start), decltype(end)>;
  index<SimdSize, IndexType> simd_i{IndexType(start)};
  constexpr auto endOffset = residualLoopPolicy == ScalarResidualLoop ? 1 : SimdSize;
  for (; simd_i.index_ + SimdSize < end + endOffset; simd_i.index_ += SimdSize)
  {
    if constexpr (sizeof...(Args) == 0)
    {
      fn(simd_i);
    }
    else
    {
      fn.template operator()<Args...>(simd_i);
    }
  }
  if constexpr (residualLoopPolicy == ScalarResidualLoop)
  {
    for (IndexType i = simd_i.index_; i < end; ++i)
    {
      if constexpr (sizeof...(Args) == 0)
      {
        fn(i);
      }
      else
      {
        fn.template operator()<Args...>(i);
      }
    }
  }
}

/**
 * Simd-ized iteration over a function using indirect indexing. The function is first called with an index_array
 * and the remainder loop is called with an integral index.
 * @tparam SimdSize Vector size.
 * @tparam Args Optional additional template arguments passed to the function call operator.
 * @tparam IteratorType Deduced type of the random access iterator defining the range of indices.
 * @param start Inclusive start of the range of indices.
 * @param end Exclusive end of the range of indices.
 * @param fn Generic function to be called. Takes one argument, whose type is either
 *   `index_array<SimdSize, IteratorType>` or `IntegralType`.
 * @param residualLoopPolicy Determines the execution policy of residual iterations. If `ScalarResidualLoop`, residual
 *   iterations are executed one by one. If `VectorResidualLoop`, residual iterations are executed vectorized. In that
 *   case the user is responsible for the handling of indices possbily extending the valid iteration range. Defaults
 *   to `ScalarResidualLoop`.
 */
template<int SimdSize, auto ... Args, std::random_access_iterator IteratorType,
  typename ResidualLoopPolicyType = ScalarResidualLoopT>
inline void loop(IteratorType start, const IteratorType& end, auto&& fn,
  ResidualLoopPolicyType residualLoopPolicy = ScalarResidualLoop)
{
  index_array<SimdSize, IteratorType> simd_i{start};
  size_t i = 0, i_end = end - start;
  constexpr auto endOffset = residualLoopPolicy == ScalarResidualLoop ? 1 : SimdSize;
  for (; i + SimdSize < i_end + endOffset; i += SimdSize, simd_i.index_ += SimdSize)
  {
    if constexpr (sizeof...(Args) == 0)
    {
      fn(simd_i);
    }
    else
    {
      fn.template operator()<Args...>(simd_i);
    }
  }
  if constexpr (residualLoopPolicy == ScalarResidualLoop)
  {
    for (; i < i_end; ++i)
    {
      if constexpr (sizeof...(Args) == 0)
      {
        fn(*(start + i));
      }
      else
      {
        fn.template operator()<Args...>(*(start + i));
      }
    }
  }
}

/**
 * Simd-ized iteration over a function using indirect indexing. The function is first called with an index_array
 * and the remainder loop is called with an integral index.
 * @tparam SimdSize Vector size.
 * @tparam Args Optional additional template arguments passed to the function call operator.
 * @tparam IteratorType Deduced type of the random access iterator defining the range of indices.
 * @param start Inclusive start of the range of indices.
 * @param end Exclusive end of the range of indices.
 * @param fn Generic function to be called. Takes two arguments. The first is the linear index starting at 0, its
 *   type is either `index<SimdSize, size_t>` or `size_t`. The second argument is the indirect index, its type is
 *   either `index_array<SimdSize, IteratorType>` or `IntegralType`.
 * @param residualLoopPolicy Determines the execution policy of residual iterations. If `ScalarResidualLoop`, residual
 *   iterations are executed one by one. If `VectorResidualLoop`, residual iterations are executed vectorized. In that
 *   case the user is responsible for the handling of indices possbily extending the valid iteration range. Defaults
 *   to `ScalarResidualLoop`.
 */
template<int SimdSize, auto ... Args, std::random_access_iterator IteratorType,
  typename ResidualLoopPolicyType = ScalarResidualLoopT>
inline void loop_with_linear_index(IteratorType start, const IteratorType& end, auto&& fn,
  ResidualLoopPolicyType residualLoopPolicy = ScalarResidualLoop)
{
  index_array<SimdSize, IteratorType> simd_i{start};
  size_t i_end = end - start;
  constexpr auto endOffset = residualLoopPolicy == ScalarResidualLoop ? 1 : SimdSize;
  index<SimdSize, size_t> i{0};
  for (; i.index_ + SimdSize < i_end + endOffset; i.index_ += SimdSize, simd_i.index_ += SimdSize)
  {
    if constexpr (sizeof...(Args) == 0)
    {
      fn(i, simd_i);
    }
    else
    {
      fn.template operator()<Args...>(i, simd_i);
    }
  }
  if constexpr (residualLoopPolicy == ScalarResidualLoop)
  {
    for (; i.index_ < i_end; ++i.index_)
    {
      if constexpr (sizeof...(Args) == 0)
      {
        fn(i.index_, *(start + i.index_));
      }
      else
      {
        fn.template operator()<Args...>(i.index_, *(start + i.index_));
      }
    }
  }
}

} //namespace simd_access

#endif //SIMD_ACCESS_LOOP

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Functions to support structure-of-simd layout
 *
 * The reflection API of simd_access enables the user to handle structures of simd variables. These so-called
 * structure-of-simd variables mimic their scalar counterpart. The user must provide two functions:
 * 1. `simdized_value` takes a scalar variable and returns a simdized (and potentially initialized)
 *   variable of the same type.
 * 2. `simd_members` takes two scalar or simdized variables and iterates over all simdized members calling a functor.
 */

#ifndef SIMD_REFLECTION
#define SIMD_REFLECTION

#include <utility>
#include <vector>

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <array>
#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

template<class T, int SimdSize>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

template<class PotentialSimdType>
concept is_stdx_simd =
  requires(PotentialSimdType x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept is_simd =
  is_stdx_simd<PotentialSimdType> ||
  requires(PotentialSimdType x) { []<class T, int SimdSize>(universal_simd<T, SimdSize>&){}(x); };

template<class T, int SimdSize>
struct auto_simd
{
  using type = universal_simd<T, SimdSize>;
};

template<simd_arithmetic T, int SimdSize>
struct auto_simd<T, SimdSize>
{
  using type = stdx::fixed_size_simd<T, SimdSize>;
};

template<class T, int SimdSize>
using auto_simd_t = typename auto_simd<T, SimdSize>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

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

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes representing simd indices to either a consecutive sequence of elements or indirect indexed elements.
 */

#ifndef SIMD_ACCESS_INDEX
#define SIMD_ACCESS_INDEX

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <array>
#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

template<class T, int SimdSize>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

template<class PotentialSimdType>
concept is_stdx_simd =
  requires(PotentialSimdType x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept is_simd =
  is_stdx_simd<PotentialSimdType> ||
  requires(PotentialSimdType x) { []<class T, int SimdSize>(universal_simd<T, SimdSize>&){}(x); };

template<class T, int SimdSize>
struct auto_simd
{
  using type = universal_simd<T, SimdSize>;
};

template<simd_arithmetic T, int SimdSize>
struct auto_simd<T, SimdSize>
{
  using type = stdx::fixed_size_simd<T, SimdSize>;
};

template<class T, int SimdSize>
using auto_simd_t = typename auto_simd<T, SimdSize>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

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

#include <type_traits>

namespace simd_access
{

template<class Location, size_t ElementSize>
class value_access;

/// Class representing a simd index to a consecutive sequence of elements.
/**
 * @tparam SimdSize Length of the simd sequence.
 * @tparam IndexType Type of the index.
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

/// Class representing a simd index to indirect indexed elements in an array.
/**
 * @tparam SimdSize Length of the simd sequence.
 * @tparam ArrayType Type of the array, which stores the indices. Defaults to std::array<size_t, SimdSize>, but could
 *   be stdx::fixed_size_simd<size_t, SimdSize>. Also size_t can be exchanged for another integral type.
 *   It is also possible to specify e.g. `int*` and thus use a pointer to a location in a larger array.
 */
template<int SimdSize, class ArrayType = std::array<size_t, SimdSize>>
struct index_array
{
  /// Return the length of the simd sequence.
  /**
   * @return The length of the simd sequence.
   */
  static constexpr int size() { return SimdSize; }

  /// Return the scalar index of a vector lane.
  /**
   * @param i Index in the vector must be in the range [0, SimdSize) .
   * @return The scalar index at vector lane i, i.e. index_[i].
   */
  auto scalar_index(int i) const { return index_[i]; }

  /// The index array, the n'th entry defines the index of the n'th element in the simd type.
  ArrayType index_;

  /// A reverse overloaded operator[] for simdized array accesses, since global operator[] is not allowed (yet).
  /**
   * @tparam T Data type of the elements in the array.
   * @param data Pointer to the array.
   * @return A value_access representing a simd access expression to indirect indexed elements in an array.
   */
  template<class T>
  auto operator[](T* data) const
  {
    return value_access<indexed_location<T, SimdSize, ArrayType>, sizeof(T)>
      (indexed_location<T, SimdSize, ArrayType>{data, index_});
  }
};

template<class PotentialIndexType>
concept is_index =
  (is_stdx_simd<PotentialIndexType> && std::is_integral_v<typename PotentialIndexType::value_type>) ||
  requires(PotentialIndexType x) { []<int SimdSize, class IndexType>(index<SimdSize, IndexType>&){}(x); } ||
  requires(PotentialIndexType x) { []<int SimdSize, class ArrayType>(index_array<SimdSize, ArrayType>&){}(x); };

/// TODO: Introduce masked_index and masked_index_array to support e.g. residual masked loops.

template<int SimdSize, class IndexType>
inline auto get_index(const index<SimdSize, IndexType>& idx, auto i)
{
  return idx.index_ + i;
}

template<int SimdSize, class ArrayType>
inline auto get_index(const index_array<SimdSize, ArrayType>& idx, auto i)
{
  return idx.scalar_index(i);
}

template<class IndexType, class Abi>
inline auto get_index(const stdx::simd<IndexType, Abi>& idx, auto i)
{
  return idx[i];
}

} //namespace simd_access

#endif //SIMD_ACCESS_INDEX

namespace simd_access
{

// Forward declarations
template<class MASK, class T>
struct where_expression;

template<class FN, simd_arithmetic DestType, simd_arithmetic SrcType>
inline void simd_members(DestType& d, const SrcType& s, FN&& func)
{
  func(d, s);
}

template<class FN, is_stdx_simd DestType, is_stdx_simd SrcType>
inline void simd_members(DestType& d, const SrcType& s, FN&& func)
{
  func(d, s);
}

template<class FN, is_stdx_simd DestType>
inline void simd_members(DestType& d, const typename DestType::value_type& s, FN&& func)
{
  func(d, s);
}

template<class FN, is_stdx_simd SrcType>
inline void simd_members(typename SrcType::value_type& d, const SrcType& s, FN&& func)
{
  func(d, s);
}

template<int SimdSize, simd_arithmetic T>
inline auto simdized_value(T)
{
  return stdx::fixed_size_simd<T, SimdSize>();
}

// overloads for std types, which can't be added after the template definition, since ADL wouldn't found it
template<int SimdSize, class T>
inline auto simdized_value(const std::vector<T>& v)
{
  std::vector<decltype(simdized_value<SimdSize>(std::declval<T>()))> result(v.size());
  return result;
}

template<class DestType, class SrcType, class FN>
inline void simd_members(std::vector<DestType>& d, const std::vector<SrcType>& s, FN&& func)
{
  for (decltype(d.size()) i = 0, e = d.size(); i < e; ++i)
  {
    simd_members(d[i], s[i], func);
  }
}

template<int SimdSize, class T, class U>
inline auto simdized_value(const std::pair<T, U>& v)
{
  return std::make_pair(simdized_value<SimdSize>(v.first), simdized_value<SimdSize>(v.second));
}

template<class DestType1, class DestType2, class SrcType1, class SrcType2, class FN>
inline void simd_members(std::pair<DestType1, DestType2>& d, const std::pair<SrcType1, SrcType2>& s, FN&& func)
{
  simd_members(d.first, s.first, func);
  simd_members(d.second, s.second, func);
}

/**
 * Loads a structure-of-simd value from a memory location defined by a base address and an linear index. The simd
 * elements to be loaded are located at the positions base, base+ElementSize, base+2*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of the scalar structure, of which `SimdSize` number of objects will be combined in a
 *   structure-of-simd.
 * @tparam SimdSize Deduced vector size of the simd type.
 * @param location Address of the memory location, at which the first scalar element is stored.
 * @return A simd value.
 */
template<size_t ElementSize, class T, int SimdSize>
  requires (!simd_arithmetic<T>)
inline auto load(const linear_location<T, SimdSize>& location)
{
  auto result = simdized_value<SimdSize>(*location.base_);
  simd_members(result, *location.base_, [&](auto&& dest, auto&& src)
    {
      dest = load<ElementSize>(linear_location<std::remove_reference_t<decltype(src)>, SimdSize>(&src));
    });
  return result;
}

/**
 * Creates a simd value from rvalues returned by the functor `subobject` applied to the range of elements defined by
 * the simd index `idx`.
 * @tparam BaseType Type of the scalar structure, of which `SimdSize` number of objects will be combined in a
 *   structure-of-simd.
 * @param base Base object.
 * @param idx Linear index.
 * @param subobject Functor returning the sub-object.
 * @return A simd value.
 */
template<class BaseType>
  requires (!simd_arithmetic<BaseType>)
inline auto load_rvalue(auto&& base, const auto& idx, auto&& subobject)
{
  decltype(simdized_value<idx.size()>(std::declval<BaseType>())) result;
  for (decltype(idx.size()) i = 0, e = idx.size(); i < e; ++i)
  {
    simd_members(result, subobject(base[get_index(idx, i)]), [&](auto&& dest, auto&& src)
      {
        dest[i] = src;
      });
  }
  return result;
}

/**
 * Creates a simd value from rvalues returned by the operator[] applied to `base`.
 * @tparam BaseType Type of the scalar structure, of which `SimdSize` number of objects will be combined in a
 *   structure-of-simd.
 * @param base Base object.
 * @param idx Linear index.
 * @return A simd value.
 */
template<class BaseType>
  requires (!simd_arithmetic<BaseType>)
inline auto load_rvalue(auto&& base, const auto& idx)
{
  decltype(simdized_value<idx.size()>(std::declval<BaseType>())) result;
  for (decltype(idx.size()) i = 0, e = idx.size(); i < e; ++i)
  {
    simd_members(result, base[get_index(idx, i)], [&](auto&& dest, auto&& src)
      {
        dest[i] = src;
      });
  }
  return result;
}

/**
 * Stores a structure-of-simd value to a memory location defined by a base address and an linear index. The simd
 * elements to be loaded are located at the positions base, base+ElementSize, base+2*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of the scalar structure, of which `SimdSize`number of objects are combined in a
 *   structure-of-simd.
 * @tparam SimdSize Deduced vector size of the simd type.
 * @tparam ExprType Deduced type of the source expression.
 * @param location Address of the memory location, to which the first scalar element is about to be stored.
 * @param expr The expression, whose result is stored. Must be convertible to a structure-of-simd.
 */
template<size_t ElementSize, class T, class ExprType, int SimdSize>
  requires (!simd_arithmetic<T>)
inline void store(const linear_location<T, SimdSize>& location, const ExprType& expr)
{
  const decltype(simdized_value<SimdSize>(std::declval<T>()))& source = expr;
  simd_members(*location.base_, source, [&](auto&& dest, auto&& src)
    {
      store<ElementSize>(
        linear_location<std::remove_reference_t<decltype(dest)>, SimdSize>(&dest), src);
    });
}

/**
 * Loads a structure-of-simd value from a memory location defined by a base address and an indirect index. The simd
 * elements to be loaded are stored at the positions base+indices[0]*ElementSize, base+indices[1]*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of the scalar structure, of which `SimdSize` number of objects will be combined in a
 *   structure-of-simd.
 * @tparam SimdSize Deduced vector size of the simd type.
 * @tparam ArrayType Deduced type of the array storing the indices.
 * @param location Address and indices of the memory location.
 * @return A simd value.
 */
template<size_t ElementSize, class T, int SimdSize, class IndexArray>
  requires (!simd_arithmetic<T>)
inline auto load(const indexed_location<T, SimdSize, IndexArray>& location)
{
  auto result = simdized_value<SimdSize>(*location.base_);
  simd_members(result, *location.base_, [&](auto&& dest, auto&& src)
    {
      dest = load<ElementSize>(indexed_location<std::remove_reference_t<decltype(src)>, SimdSize, IndexArray>(
        &src, location.indices_));
    });
  return result;
}

/**
 * Stores a structure-of-simd value to a memory location defined by a base address and an indirect index. The simd
 * elements are stored at the positions base+indices[0]*ElementSize, base+indices[1]*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of the scalar structure, of which `SimdSize`number of objects are combined in a
 *   structure-of-simd.
 * @tparam SimdSize Deduced vector size of the simd type.
 * @tparam ArrayType Deduced type of the array storing the indices.
 * @tparam ExprType Deduced type of the source expression.
 * @param location Address and indices of the memory location.
 * @param expr The expression, whose result is stored. Must be convertible to a structure-of-simd.
 */
template<size_t ElementSize, class T, class ExprType, int SimdSize, class IndexArray>
  requires (!simd_arithmetic<T>)
inline void store(const indexed_location<T, SimdSize, IndexArray>& location, const ExprType& expr)
{
  const decltype(simdized_value<SimdSize>(std::declval<T>()))& source = expr;
  simd_members(*location.base_, source, [&](auto&& dest, auto&& src)
    {
      store<ElementSize>(indexed_location<std::remove_reference_t<decltype(dest)>, SimdSize, IndexArray>(&dest, location.indices_), src);
    });
}

/**
 * Returns a `where_expression` for structure-of-simd types, which are unsupported by stdx::simd.
 * @tparam MASK Deduced type of the simd mask.
 * @tparam T Deduced structure-of-simd type.
 * @param mask The value of the simd mask.
 * @param dest Reference to the structure-of-simd value, to which `mask` is applied.
 */
template<class M, class T>
  requires((!simd_arithmetic<T>) && (!is_stdx_simd<T>))
inline auto where(const M& mask, T& dest)
{
  return where_expression<M, T>(mask, dest);
}

/**
 * Extends `stdx::where_expression` for structure-of-simd types.
 * @tparam M Type of the simd mask.
 * @tparam T Structure-of-simd type.
 */
template<class M, class T>
struct where_expression
{
  M mask_;
  T& destination_;

  where_expression(const M& m, T& dest) :
    mask_(m),
    destination_(dest)
  {}

  auto& operator=(const T& source) &&
  {
    simd_members(destination_, source, [&](auto& d, const auto& s)
      {
        using stdx::where;
        using simd_access::where;
        where(mask_, d) = s;
      });
    return *this;
  }
};

} //namespace simd_access

#endif //SIMD_REFLECTION

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_UNIVERSAL_SIMD
#define SIMD_ACCESS_UNIVERSAL_SIMD

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <array>
#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

template<class T, int SimdSize>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

template<class PotentialSimdType>
concept is_stdx_simd =
  requires(PotentialSimdType x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept is_simd =
  is_stdx_simd<PotentialSimdType> ||
  requires(PotentialSimdType x) { []<class T, int SimdSize>(universal_simd<T, SimdSize>&){}(x); };

template<class T, int SimdSize>
struct auto_simd
{
  using type = universal_simd<T, SimdSize>;
};

template<simd_arithmetic T, int SimdSize>
struct auto_simd<T, SimdSize>
{
  using type = stdx::fixed_size_simd<T, SimdSize>;
};

template<class T, int SimdSize>
using auto_simd_t = typename auto_simd<T, SimdSize>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes representing simd indices to either a consecutive sequence of elements or indirect indexed elements.
 */

#ifndef SIMD_ACCESS_INDEX
#define SIMD_ACCESS_INDEX

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <array>
#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

template<class T, int SimdSize>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

template<class PotentialSimdType>
concept is_stdx_simd =
  requires(PotentialSimdType x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept is_simd =
  is_stdx_simd<PotentialSimdType> ||
  requires(PotentialSimdType x) { []<class T, int SimdSize>(universal_simd<T, SimdSize>&){}(x); };

template<class T, int SimdSize>
struct auto_simd
{
  using type = universal_simd<T, SimdSize>;
};

template<simd_arithmetic T, int SimdSize>
struct auto_simd<T, SimdSize>
{
  using type = stdx::fixed_size_simd<T, SimdSize>;
};

template<class T, int SimdSize>
using auto_simd_t = typename auto_simd<T, SimdSize>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

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

#include <type_traits>

namespace simd_access
{

template<class Location, size_t ElementSize>
class value_access;

/// Class representing a simd index to a consecutive sequence of elements.
/**
 * @tparam SimdSize Length of the simd sequence.
 * @tparam IndexType Type of the index.
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

/// Class representing a simd index to indirect indexed elements in an array.
/**
 * @tparam SimdSize Length of the simd sequence.
 * @tparam ArrayType Type of the array, which stores the indices. Defaults to std::array<size_t, SimdSize>, but could
 *   be stdx::fixed_size_simd<size_t, SimdSize>. Also size_t can be exchanged for another integral type.
 *   It is also possible to specify e.g. `int*` and thus use a pointer to a location in a larger array.
 */
template<int SimdSize, class ArrayType = std::array<size_t, SimdSize>>
struct index_array
{
  /// Return the length of the simd sequence.
  /**
   * @return The length of the simd sequence.
   */
  static constexpr int size() { return SimdSize; }

  /// Return the scalar index of a vector lane.
  /**
   * @param i Index in the vector must be in the range [0, SimdSize) .
   * @return The scalar index at vector lane i, i.e. index_[i].
   */
  auto scalar_index(int i) const { return index_[i]; }

  /// The index array, the n'th entry defines the index of the n'th element in the simd type.
  ArrayType index_;

  /// A reverse overloaded operator[] for simdized array accesses, since global operator[] is not allowed (yet).
  /**
   * @tparam T Data type of the elements in the array.
   * @param data Pointer to the array.
   * @return A value_access representing a simd access expression to indirect indexed elements in an array.
   */
  template<class T>
  auto operator[](T* data) const
  {
    return value_access<indexed_location<T, SimdSize, ArrayType>, sizeof(T)>
      (indexed_location<T, SimdSize, ArrayType>{data, index_});
  }
};

template<class PotentialIndexType>
concept is_index =
  (is_stdx_simd<PotentialIndexType> && std::is_integral_v<typename PotentialIndexType::value_type>) ||
  requires(PotentialIndexType x) { []<int SimdSize, class IndexType>(index<SimdSize, IndexType>&){}(x); } ||
  requires(PotentialIndexType x) { []<int SimdSize, class ArrayType>(index_array<SimdSize, ArrayType>&){}(x); };

/// TODO: Introduce masked_index and masked_index_array to support e.g. residual masked loops.

template<int SimdSize, class IndexType>
inline auto get_index(const index<SimdSize, IndexType>& idx, auto i)
{
  return idx.index_ + i;
}

template<int SimdSize, class ArrayType>
inline auto get_index(const index_array<SimdSize, ArrayType>& idx, auto i)
{
  return idx.scalar_index(i);
}

template<class IndexType, class Abi>
inline auto get_index(const stdx::simd<IndexType, Abi>& idx, auto i)
{
  return idx[i];
}

} //namespace simd_access

#endif //SIMD_ACCESS_INDEX

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Functions to support structure-of-simd layout
 *
 * The reflection API of simd_access enables the user to handle structures of simd variables. These so-called
 * structure-of-simd variables mimic their scalar counterpart. The user must provide two functions:
 * 1. `simdized_value` takes a scalar variable and returns a simdized (and potentially initialized)
 *   variable of the same type.
 * 2. `simd_members` takes two scalar or simdized variables and iterates over all simdized members calling a functor.
 */

#ifndef SIMD_REFLECTION
#define SIMD_REFLECTION

#include <utility>
#include <vector>

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <array>
#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

template<class T, int SimdSize>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

template<class PotentialSimdType>
concept is_stdx_simd =
  requires(PotentialSimdType x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept is_simd =
  is_stdx_simd<PotentialSimdType> ||
  requires(PotentialSimdType x) { []<class T, int SimdSize>(universal_simd<T, SimdSize>&){}(x); };

template<class T, int SimdSize>
struct auto_simd
{
  using type = universal_simd<T, SimdSize>;
};

template<simd_arithmetic T, int SimdSize>
struct auto_simd<T, SimdSize>
{
  using type = stdx::fixed_size_simd<T, SimdSize>;
};

template<class T, int SimdSize>
using auto_simd_t = typename auto_simd<T, SimdSize>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

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

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes representing simd indices to either a consecutive sequence of elements or indirect indexed elements.
 */

#ifndef SIMD_ACCESS_INDEX
#define SIMD_ACCESS_INDEX

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <array>
#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

template<class T, int SimdSize>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

template<class PotentialSimdType>
concept is_stdx_simd =
  requires(PotentialSimdType x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept is_simd =
  is_stdx_simd<PotentialSimdType> ||
  requires(PotentialSimdType x) { []<class T, int SimdSize>(universal_simd<T, SimdSize>&){}(x); };

template<class T, int SimdSize>
struct auto_simd
{
  using type = universal_simd<T, SimdSize>;
};

template<simd_arithmetic T, int SimdSize>
struct auto_simd<T, SimdSize>
{
  using type = stdx::fixed_size_simd<T, SimdSize>;
};

template<class T, int SimdSize>
using auto_simd_t = typename auto_simd<T, SimdSize>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

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

#include <type_traits>

namespace simd_access
{

template<class Location, size_t ElementSize>
class value_access;

/// Class representing a simd index to a consecutive sequence of elements.
/**
 * @tparam SimdSize Length of the simd sequence.
 * @tparam IndexType Type of the index.
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

/// Class representing a simd index to indirect indexed elements in an array.
/**
 * @tparam SimdSize Length of the simd sequence.
 * @tparam ArrayType Type of the array, which stores the indices. Defaults to std::array<size_t, SimdSize>, but could
 *   be stdx::fixed_size_simd<size_t, SimdSize>. Also size_t can be exchanged for another integral type.
 *   It is also possible to specify e.g. `int*` and thus use a pointer to a location in a larger array.
 */
template<int SimdSize, class ArrayType = std::array<size_t, SimdSize>>
struct index_array
{
  /// Return the length of the simd sequence.
  /**
   * @return The length of the simd sequence.
   */
  static constexpr int size() { return SimdSize; }

  /// Return the scalar index of a vector lane.
  /**
   * @param i Index in the vector must be in the range [0, SimdSize) .
   * @return The scalar index at vector lane i, i.e. index_[i].
   */
  auto scalar_index(int i) const { return index_[i]; }

  /// The index array, the n'th entry defines the index of the n'th element in the simd type.
  ArrayType index_;

  /// A reverse overloaded operator[] for simdized array accesses, since global operator[] is not allowed (yet).
  /**
   * @tparam T Data type of the elements in the array.
   * @param data Pointer to the array.
   * @return A value_access representing a simd access expression to indirect indexed elements in an array.
   */
  template<class T>
  auto operator[](T* data) const
  {
    return value_access<indexed_location<T, SimdSize, ArrayType>, sizeof(T)>
      (indexed_location<T, SimdSize, ArrayType>{data, index_});
  }
};

template<class PotentialIndexType>
concept is_index =
  (is_stdx_simd<PotentialIndexType> && std::is_integral_v<typename PotentialIndexType::value_type>) ||
  requires(PotentialIndexType x) { []<int SimdSize, class IndexType>(index<SimdSize, IndexType>&){}(x); } ||
  requires(PotentialIndexType x) { []<int SimdSize, class ArrayType>(index_array<SimdSize, ArrayType>&){}(x); };

/// TODO: Introduce masked_index and masked_index_array to support e.g. residual masked loops.

template<int SimdSize, class IndexType>
inline auto get_index(const index<SimdSize, IndexType>& idx, auto i)
{
  return idx.index_ + i;
}

template<int SimdSize, class ArrayType>
inline auto get_index(const index_array<SimdSize, ArrayType>& idx, auto i)
{
  return idx.scalar_index(i);
}

template<class IndexType, class Abi>
inline auto get_index(const stdx::simd<IndexType, Abi>& idx, auto i)
{
  return idx[i];
}

} //namespace simd_access

#endif //SIMD_ACCESS_INDEX

namespace simd_access
{

// Forward declarations
template<class MASK, class T>
struct where_expression;

template<class FN, simd_arithmetic DestType, simd_arithmetic SrcType>
inline void simd_members(DestType& d, const SrcType& s, FN&& func)
{
  func(d, s);
}

template<class FN, is_stdx_simd DestType, is_stdx_simd SrcType>
inline void simd_members(DestType& d, const SrcType& s, FN&& func)
{
  func(d, s);
}

template<class FN, is_stdx_simd DestType>
inline void simd_members(DestType& d, const typename DestType::value_type& s, FN&& func)
{
  func(d, s);
}

template<class FN, is_stdx_simd SrcType>
inline void simd_members(typename SrcType::value_type& d, const SrcType& s, FN&& func)
{
  func(d, s);
}

template<int SimdSize, simd_arithmetic T>
inline auto simdized_value(T)
{
  return stdx::fixed_size_simd<T, SimdSize>();
}

// overloads for std types, which can't be added after the template definition, since ADL wouldn't found it
template<int SimdSize, class T>
inline auto simdized_value(const std::vector<T>& v)
{
  std::vector<decltype(simdized_value<SimdSize>(std::declval<T>()))> result(v.size());
  return result;
}

template<class DestType, class SrcType, class FN>
inline void simd_members(std::vector<DestType>& d, const std::vector<SrcType>& s, FN&& func)
{
  for (decltype(d.size()) i = 0, e = d.size(); i < e; ++i)
  {
    simd_members(d[i], s[i], func);
  }
}

template<int SimdSize, class T, class U>
inline auto simdized_value(const std::pair<T, U>& v)
{
  return std::make_pair(simdized_value<SimdSize>(v.first), simdized_value<SimdSize>(v.second));
}

template<class DestType1, class DestType2, class SrcType1, class SrcType2, class FN>
inline void simd_members(std::pair<DestType1, DestType2>& d, const std::pair<SrcType1, SrcType2>& s, FN&& func)
{
  simd_members(d.first, s.first, func);
  simd_members(d.second, s.second, func);
}

/**
 * Loads a structure-of-simd value from a memory location defined by a base address and an linear index. The simd
 * elements to be loaded are located at the positions base, base+ElementSize, base+2*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of the scalar structure, of which `SimdSize` number of objects will be combined in a
 *   structure-of-simd.
 * @tparam SimdSize Deduced vector size of the simd type.
 * @param location Address of the memory location, at which the first scalar element is stored.
 * @return A simd value.
 */
template<size_t ElementSize, class T, int SimdSize>
  requires (!simd_arithmetic<T>)
inline auto load(const linear_location<T, SimdSize>& location)
{
  auto result = simdized_value<SimdSize>(*location.base_);
  simd_members(result, *location.base_, [&](auto&& dest, auto&& src)
    {
      dest = load<ElementSize>(linear_location<std::remove_reference_t<decltype(src)>, SimdSize>(&src));
    });
  return result;
}

/**
 * Creates a simd value from rvalues returned by the functor `subobject` applied to the range of elements defined by
 * the simd index `idx`.
 * @tparam BaseType Type of the scalar structure, of which `SimdSize` number of objects will be combined in a
 *   structure-of-simd.
 * @param base Base object.
 * @param idx Linear index.
 * @param subobject Functor returning the sub-object.
 * @return A simd value.
 */
template<class BaseType>
  requires (!simd_arithmetic<BaseType>)
inline auto load_rvalue(auto&& base, const auto& idx, auto&& subobject)
{
  decltype(simdized_value<idx.size()>(std::declval<BaseType>())) result;
  for (decltype(idx.size()) i = 0, e = idx.size(); i < e; ++i)
  {
    simd_members(result, subobject(base[get_index(idx, i)]), [&](auto&& dest, auto&& src)
      {
        dest[i] = src;
      });
  }
  return result;
}

/**
 * Creates a simd value from rvalues returned by the operator[] applied to `base`.
 * @tparam BaseType Type of the scalar structure, of which `SimdSize` number of objects will be combined in a
 *   structure-of-simd.
 * @param base Base object.
 * @param idx Linear index.
 * @return A simd value.
 */
template<class BaseType>
  requires (!simd_arithmetic<BaseType>)
inline auto load_rvalue(auto&& base, const auto& idx)
{
  decltype(simdized_value<idx.size()>(std::declval<BaseType>())) result;
  for (decltype(idx.size()) i = 0, e = idx.size(); i < e; ++i)
  {
    simd_members(result, base[get_index(idx, i)], [&](auto&& dest, auto&& src)
      {
        dest[i] = src;
      });
  }
  return result;
}

/**
 * Stores a structure-of-simd value to a memory location defined by a base address and an linear index. The simd
 * elements to be loaded are located at the positions base, base+ElementSize, base+2*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of the scalar structure, of which `SimdSize`number of objects are combined in a
 *   structure-of-simd.
 * @tparam SimdSize Deduced vector size of the simd type.
 * @tparam ExprType Deduced type of the source expression.
 * @param location Address of the memory location, to which the first scalar element is about to be stored.
 * @param expr The expression, whose result is stored. Must be convertible to a structure-of-simd.
 */
template<size_t ElementSize, class T, class ExprType, int SimdSize>
  requires (!simd_arithmetic<T>)
inline void store(const linear_location<T, SimdSize>& location, const ExprType& expr)
{
  const decltype(simdized_value<SimdSize>(std::declval<T>()))& source = expr;
  simd_members(*location.base_, source, [&](auto&& dest, auto&& src)
    {
      store<ElementSize>(
        linear_location<std::remove_reference_t<decltype(dest)>, SimdSize>(&dest), src);
    });
}

/**
 * Loads a structure-of-simd value from a memory location defined by a base address and an indirect index. The simd
 * elements to be loaded are stored at the positions base+indices[0]*ElementSize, base+indices[1]*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of the scalar structure, of which `SimdSize` number of objects will be combined in a
 *   structure-of-simd.
 * @tparam SimdSize Deduced vector size of the simd type.
 * @tparam ArrayType Deduced type of the array storing the indices.
 * @param location Address and indices of the memory location.
 * @return A simd value.
 */
template<size_t ElementSize, class T, int SimdSize, class IndexArray>
  requires (!simd_arithmetic<T>)
inline auto load(const indexed_location<T, SimdSize, IndexArray>& location)
{
  auto result = simdized_value<SimdSize>(*location.base_);
  simd_members(result, *location.base_, [&](auto&& dest, auto&& src)
    {
      dest = load<ElementSize>(indexed_location<std::remove_reference_t<decltype(src)>, SimdSize, IndexArray>(
        &src, location.indices_));
    });
  return result;
}

/**
 * Stores a structure-of-simd value to a memory location defined by a base address and an indirect index. The simd
 * elements are stored at the positions base+indices[0]*ElementSize, base+indices[1]*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of the scalar structure, of which `SimdSize`number of objects are combined in a
 *   structure-of-simd.
 * @tparam SimdSize Deduced vector size of the simd type.
 * @tparam ArrayType Deduced type of the array storing the indices.
 * @tparam ExprType Deduced type of the source expression.
 * @param location Address and indices of the memory location.
 * @param expr The expression, whose result is stored. Must be convertible to a structure-of-simd.
 */
template<size_t ElementSize, class T, class ExprType, int SimdSize, class IndexArray>
  requires (!simd_arithmetic<T>)
inline void store(const indexed_location<T, SimdSize, IndexArray>& location, const ExprType& expr)
{
  const decltype(simdized_value<SimdSize>(std::declval<T>()))& source = expr;
  simd_members(*location.base_, source, [&](auto&& dest, auto&& src)
    {
      store<ElementSize>(indexed_location<std::remove_reference_t<decltype(dest)>, SimdSize, IndexArray>(&dest, location.indices_), src);
    });
}

/**
 * Returns a `where_expression` for structure-of-simd types, which are unsupported by stdx::simd.
 * @tparam MASK Deduced type of the simd mask.
 * @tparam T Deduced structure-of-simd type.
 * @param mask The value of the simd mask.
 * @param dest Reference to the structure-of-simd value, to which `mask` is applied.
 */
template<class M, class T>
  requires((!simd_arithmetic<T>) && (!is_stdx_simd<T>))
inline auto where(const M& mask, T& dest)
{
  return where_expression<M, T>(mask, dest);
}

/**
 * Extends `stdx::where_expression` for structure-of-simd types.
 * @tparam M Type of the simd mask.
 * @tparam T Structure-of-simd type.
 */
template<class M, class T>
struct where_expression
{
  M mask_;
  T& destination_;

  where_expression(const M& m, T& dest) :
    mask_(m),
    destination_(dest)
  {}

  auto& operator=(const T& source) &&
  {
    simd_members(destination_, source, [&](auto& d, const auto& s)
      {
        using stdx::where;
        using simd_access::where;
        where(mask_, d) = s;
      });
    return *this;
  }
};

} //namespace simd_access

#endif //SIMD_REFLECTION

#include <functional>
#include <type_traits>

namespace simd_access
{

template<class T, int SimdSize>
struct universal_simd : std::array<T, SimdSize>
{
  static constexpr auto size() { return SimdSize; }

  template<class G>
  explicit universal_simd(G&& generator) :
    std::array<T, SimdSize>([&]<int... I>(std::integer_sequence<int, I...>) -> std::array<T, SimdSize>
      {
        return {{ generator(std::integral_constant<int, I>())... }};
      } (std::make_integer_sequence<int, SimdSize>())) {}

  universal_simd() = default;
};

inline decltype(auto) generate_universal(std::integral auto i, auto&& generator)
{
  return generator(i);
}

template<class IndexType>
  requires(!std::integral<IndexType>)
inline decltype(auto) generate_universal(const IndexType& idx, auto&& generator)
{
  return universal_simd<decltype(generator(0)), idx.size()>([&](auto i) { return generator(get_index(idx, i)); });
}

template<class T, class Func>
  requires(!is_simd<T>)
inline decltype(auto) universal_access(const T& v, Func&& subobject)
{
  return subobject(v);
}

template<class T, int SimdSize, class Func>
inline auto universal_access(const simd_access::universal_simd<T, SimdSize>& v, Func&& subobject)
{
  decltype(simdized_value<SimdSize>(std::declval<decltype(subobject(static_cast<const std::unwrap_reference_t<T>&>(v[0])))>())) result;
  for (int i = 0; i < SimdSize; ++i)
  {
    simd_members(result, subobject(static_cast<const std::unwrap_reference_t<T>&>(v[i])), [&](auto&& d, auto&& s) { d[i] = s; });
  }
  return result;
}

#define SIMD_UNIVERSAL_ACCESS(value, expr) \
  simd_access::universal_access(value, [&](auto&& element) { return element expr; })

} //namespace simd_access

#endif //SIMD_ACCESS_UNIVERSAL_SIMD

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Class for accessing simd values via simd indices.
 */

#ifndef SIMD_ACCESS_VALUE_ACCESS
#define SIMD_ACCESS_VALUE_ACCESS

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Base class for operator overloads
 * This file demonstrates a possible approach to overload the member access operator.
 */

#ifndef SIMD_ACCESS_OPERATOR_OVERLOAD
#define SIMD_ACCESS_OPERATOR_OVERLOAD

#include <type_traits>
// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <array>
#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

template<class T, int SimdSize>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

template<class PotentialSimdType>
concept is_stdx_simd =
  requires(PotentialSimdType x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept is_simd =
  is_stdx_simd<PotentialSimdType> ||
  requires(PotentialSimdType x) { []<class T, int SimdSize>(universal_simd<T, SimdSize>&){}(x); };

template<class T, int SimdSize>
struct auto_simd
{
  using type = universal_simd<T, SimdSize>;
};

template<simd_arithmetic T, int SimdSize>
struct auto_simd<T, SimdSize>
{
  using type = stdx::fixed_size_simd<T, SimdSize>;
};

template<class T, int SimdSize>
using auto_simd_t = typename auto_simd<T, SimdSize>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

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

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Global functions to load and store simd values to using several addressing modes.
 */

#ifndef SIMD_LOAD_STORE
#define SIMD_LOAD_STORE

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <array>
#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

template<class T, int SimdSize>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

template<class PotentialSimdType>
concept is_stdx_simd =
  requires(PotentialSimdType x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept is_simd =
  is_stdx_simd<PotentialSimdType> ||
  requires(PotentialSimdType x) { []<class T, int SimdSize>(universal_simd<T, SimdSize>&){}(x); };

template<class T, int SimdSize>
struct auto_simd
{
  using type = universal_simd<T, SimdSize>;
};

template<simd_arithmetic T, int SimdSize>
struct auto_simd<T, SimdSize>
{
  using type = stdx::fixed_size_simd<T, SimdSize>;
};

template<class T, int SimdSize>
using auto_simd_t = typename auto_simd<T, SimdSize>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

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

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes representing simd indices to either a consecutive sequence of elements or indirect indexed elements.
 */

#ifndef SIMD_ACCESS_INDEX
#define SIMD_ACCESS_INDEX

// See the file "LICENSE" for the full license governing this code.

/**
 * @file
 * @brief Classes defining base types and concepts.
 */

#ifndef SIMD_ACCESS_BASE
#define SIMD_ACCESS_BASE

#include <array>
#include <experimental/simd>
#include <type_traits>
namespace stdx = std::experimental;

namespace simd_access
{

template<class T, int SimdSize>
struct universal_simd;

template<typename T>
concept simd_arithmetic =
  std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

template<class PotentialSimdType>
concept is_stdx_simd =
  requires(PotentialSimdType x) { []<class T, class Abi>(stdx::simd<T, Abi>&){}(x); };

template<class PotentialSimdType>
concept is_simd =
  is_stdx_simd<PotentialSimdType> ||
  requires(PotentialSimdType x) { []<class T, int SimdSize>(universal_simd<T, SimdSize>&){}(x); };

template<class T, int SimdSize>
struct auto_simd
{
  using type = universal_simd<T, SimdSize>;
};

template<simd_arithmetic T, int SimdSize>
struct auto_simd<T, SimdSize>
{
  using type = stdx::fixed_size_simd<T, SimdSize>;
};

template<class T, int SimdSize>
using auto_simd_t = typename auto_simd<T, SimdSize>::type;

} //namespace simd_access

#endif //SIMD_ACCESS_BASE

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

#include <type_traits>

namespace simd_access
{

template<class Location, size_t ElementSize>
class value_access;

/// Class representing a simd index to a consecutive sequence of elements.
/**
 * @tparam SimdSize Length of the simd sequence.
 * @tparam IndexType Type of the index.
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

/// Class representing a simd index to indirect indexed elements in an array.
/**
 * @tparam SimdSize Length of the simd sequence.
 * @tparam ArrayType Type of the array, which stores the indices. Defaults to std::array<size_t, SimdSize>, but could
 *   be stdx::fixed_size_simd<size_t, SimdSize>. Also size_t can be exchanged for another integral type.
 *   It is also possible to specify e.g. `int*` and thus use a pointer to a location in a larger array.
 */
template<int SimdSize, class ArrayType = std::array<size_t, SimdSize>>
struct index_array
{
  /// Return the length of the simd sequence.
  /**
   * @return The length of the simd sequence.
   */
  static constexpr int size() { return SimdSize; }

  /// Return the scalar index of a vector lane.
  /**
   * @param i Index in the vector must be in the range [0, SimdSize) .
   * @return The scalar index at vector lane i, i.e. index_[i].
   */
  auto scalar_index(int i) const { return index_[i]; }

  /// The index array, the n'th entry defines the index of the n'th element in the simd type.
  ArrayType index_;

  /// A reverse overloaded operator[] for simdized array accesses, since global operator[] is not allowed (yet).
  /**
   * @tparam T Data type of the elements in the array.
   * @param data Pointer to the array.
   * @return A value_access representing a simd access expression to indirect indexed elements in an array.
   */
  template<class T>
  auto operator[](T* data) const
  {
    return value_access<indexed_location<T, SimdSize, ArrayType>, sizeof(T)>
      (indexed_location<T, SimdSize, ArrayType>{data, index_});
  }
};

template<class PotentialIndexType>
concept is_index =
  (is_stdx_simd<PotentialIndexType> && std::is_integral_v<typename PotentialIndexType::value_type>) ||
  requires(PotentialIndexType x) { []<int SimdSize, class IndexType>(index<SimdSize, IndexType>&){}(x); } ||
  requires(PotentialIndexType x) { []<int SimdSize, class ArrayType>(index_array<SimdSize, ArrayType>&){}(x); };

/// TODO: Introduce masked_index and masked_index_array to support e.g. residual masked loops.

template<int SimdSize, class IndexType>
inline auto get_index(const index<SimdSize, IndexType>& idx, auto i)
{
  return idx.index_ + i;
}

template<int SimdSize, class ArrayType>
inline auto get_index(const index_array<SimdSize, ArrayType>& idx, auto i)
{
  return idx.scalar_index(i);
}

template<class IndexType, class Abi>
inline auto get_index(const stdx::simd<IndexType, Abi>& idx, auto i)
{
  return idx[i];
}

} //namespace simd_access

#endif //SIMD_ACCESS_INDEX

namespace simd_access
{

/**
 * Stores a simd value to a memory location defined by a base address and an linear index. The simd elements are
 * stored at the positions base, base+ElementSize, base+2*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of a simd element.
 * @tparam SimdSize Deduced vector size of the simd type.
 * @param location Address of the memory location, at which the first simd element is stored.
 * @param source Simd value to be stored.
 */
template<size_t ElementSize, simd_arithmetic T, int SimdSize>
inline void store(const linear_location<T, SimdSize>& location, const stdx::fixed_size_simd<T, SimdSize>& source)
{
  if constexpr (sizeof(T) == ElementSize)
  {
    source.copy_to(location.base_, stdx::element_aligned);
  }
  else
  {
    // scatter with constant pitch
    for (int i = 0; i < SimdSize; ++i)
    {
      *reinterpret_cast<T*>(reinterpret_cast<char*>(location.base_) + ElementSize * i) = source[i];
    }
  }
}

/**
 * Stores a simd value to a memory location defined by a base address and an indirect index. The simd elements are
 * stored at the positions base+indices[0]*ElementSize, base+indices[1]*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of a simd element.
 * @tparam SimdSize Deduced vector size of the simd type.
 * @tparam ArrayType Deduced type of the array storing the indices.
 * @param location Address and indices of the memory location.
 * @param source Simd value to be stored.
 */
template<size_t ElementSize, simd_arithmetic T, int SimdSize, class ArrayType>
inline void store(const indexed_location<T, SimdSize, ArrayType>& location,
  const stdx::fixed_size_simd<T, SimdSize>& source)
{
  // scatter with indirect indices
  for (int i = 0; i < SimdSize; ++i)
  {
    *reinterpret_cast<T*>(reinterpret_cast<char*>(location.base_) + ElementSize * location.indices_[i]) = source[i];
  }
}

/**
 * Loads a simd value from a memory location defined by a base address and an linear index. The simd elements to be
 * loaded are located at the positions base, base+ElementSize, base+2*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Type of a simd element.
 * @tparam SimdSize Vector size of the simd type.
 * @param location Address of the memory location, at which the first scalar element is stored.
 * @return A simd value.
 */
template<size_t ElementSize, simd_arithmetic T, int SimdSize>
inline auto load(const linear_location<T, SimdSize>& location)
{
  using ResultType = stdx::fixed_size_simd<std::remove_const_t<T>, SimdSize>;
  if constexpr (sizeof(T) == ElementSize)
  {
    return ResultType(location.base_, stdx::element_aligned);
  }
  else
  {
    // gather with constant pitch
    return ResultType([&](int i)
      {
        return *reinterpret_cast<const T*>(reinterpret_cast<const char*>(location.base_) + ElementSize * i);
      });
  }
}

/**
 * Loads a simd value from a memory location defined by a base address and an indirect index. The simd elements to be
 * loaded are stored at the positions base+indices[0]*ElementSize, base+indices[1]*ElementSize, ...
 * @tparam ElementSize Size in bytes of the type of the simd-indexed element.
 * @tparam T Deduced type of a simd element.
 * @tparam SimdSize Deduced vector size of the simd type.
 * @tparam ArrayType Deduced type of the array storing the indices.
 * @param location Address and indices of the memory location.
 * @return A simd value.
 */
template<size_t ElementSize, simd_arithmetic T, int SimdSize, class ArrayType>
inline auto load(const indexed_location<T, SimdSize, ArrayType>& location)
{
  // gather with indirect indices
  return stdx::fixed_size_simd<std::remove_const_t<T>, SimdSize>([&](int i)
    {
      return *reinterpret_cast<const T*>
        (reinterpret_cast<const char*>(location.base_) + ElementSize * location.indices_[i]);
    });
}

/**
 * Creates a simd value from rvalues returned by the operator[] applied to `base`.
 * @tparam BaseType Type of an simd element.
 * @param base Base object.
 * @param idx Index.
 * @return A simd value.
 */
template<simd_arithmetic BaseType>
inline auto load_rvalue(auto&& base, const auto& idx)
{
  return stdx::fixed_size_simd<BaseType, idx.size()>([&](auto i) { return base[get_index(idx, i)]; });
}

/**
 * Creates a simd value from rvalues returned by the functor `subobject` applied to the range of elements defined by
 * the simd index `idx`.
 * @tparam BaseType Type of an simd element.
 * @param base Base object.
 * @param idx Index.
 * @param subobject Functor returning the sub-object.
 * @return A simd value.
 */
template<simd_arithmetic BaseType>
inline auto load_rvalue(auto&& base, const auto& idx, auto&& subobject)
{
  return stdx::fixed_size_simd<BaseType, idx.size()>([&](auto i) { return subobject(base[get_index(idx, i)]); });
}

} //namespace simd_access

#endif //SIMD_LOAD_STORE

namespace simd_access
{

#define VALUE_ACCESS_BIN_OP( op ) \
  auto operator op(const auto& source) { return to_simd() op source; }

#define VALUE_ACCESS_BIN_ASSIGNMENT_OP( op ) \
  void operator op##=(const auto& source) && { store<ElementSize>(location_, to_simd() op source); }

#define VALUE_ACCESS_MEMBER_OPS( op ) \
  VALUE_ACCESS_BIN_OP( op ) \
  VALUE_ACCESS_BIN_ASSIGNMENT_OP( op )

#define VALUE_ACCESS_SCALAR_BIN_OP( op ) \
  template<class Location, size_t ElementSize> \
  inline auto operator op(const auto& o1, const value_access<Location, ElementSize>& o2) \
  { \
    return o1 op o2.to_simd(); \
  }

/// Class representing a simd-access (read or write) to a memory location.
/**
 * @tparam Location Type of the location of the simd data.
 * @tparam ElementSize Size of the array elements, which (or one of its members) are accessed by the simd index.
 */
template<class Location, size_t ElementSize = 0>
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

  /// Implementation of overloaded member operator, i.e. operator.()
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
  value_access(const Location& location) :
    location_(location)
  {}

private:
  Location location_;
};

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

namespace simd_access
{

template<bool isLvalue>
struct LValueSeparator;

template<>
struct LValueSeparator<true>
{
  static decltype(auto) to_simd(auto&& base, std::integral auto i)
  {
    return base[i];
  }

  static decltype(auto) to_simd(auto&& base, std::integral auto i, auto&& subobject)
  {
    return subobject(base[i]);
  }

  template<int SimdSize, class IndexType>
  static auto get_base_address(auto&& base_addr, const index<SimdSize, IndexType>& i)
  {
    return &base_addr[i.index_];
  }

  template<class IndexType, class Abi>
  static auto get_base_address(auto&& base_addr, const stdx::simd<IndexType, Abi>&)
  {
    return &base_addr[0];
  }

  template<int SimdSize, class ArrayType>
  static auto get_base_address(auto&& base_addr, const index_array<SimdSize, ArrayType>&)
  {
    return &base_addr[0];
  }

  template<int SimdSize, class IndexType>
  static auto get_base_address(auto&& base_addr, const index<SimdSize, IndexType>& i, auto&& subobject)
  {
    return &subobject(base_addr[i.index_]);
  }

  template<class IndexType, class Abi>
  static auto get_base_address(auto&& base_addr, const stdx::simd<IndexType, Abi>&, auto&& subobject)
  {
    return &subobject(base_addr[0]);
  }

  template<int SimdSize, class ArrayType>
  static auto get_base_address(auto&& base_addr, const index_array<SimdSize, ArrayType>&, auto&& subobject)
  {
    return &subobject(base_addr[0]);
  }

  template<size_t ElementSize, class T, int SimdSize, class IndexType>
  static auto get_direct_value_access(T* base, const index<SimdSize, IndexType>&)
  {
    return make_value_access<ElementSize>(linear_location<T, SimdSize>{base});
  }

  template<size_t ElementSize, class T, int SimdSize, class ArrayType>
  static auto get_direct_value_access(T* base, const index_array<SimdSize, ArrayType>& idx)
  {
    return make_value_access<ElementSize>(indexed_location<T, SimdSize, ArrayType>{base, idx.index_});
  }

  template<size_t ElementSize, class T, class IndexType, class Abi>
  static auto get_direct_value_access(T* base, const stdx::simd<IndexType, Abi>& idx)
  {
    return make_value_access<ElementSize>(indexed_location<T, idx.size(), stdx::simd<IndexType, Abi>>{base, idx});
  }

  template<class IndexType, class... Func>
    requires(!std::integral<IndexType>)
  static auto to_simd(auto&& base, const IndexType& indices, Func&&... subobject)
  {
    return get_direct_value_access<sizeof(decltype(base[0]))>(get_base_address(base, indices, subobject...), indices);
  }
};

template<>
struct LValueSeparator<false>
{
  static decltype(auto) to_simd(auto&& base, std::integral auto i)
  {
    return base[i];
  }

  static decltype(auto) to_simd(auto&& base, std::integral auto i, auto&& subobject)
  {
    return subobject(base[i]);
  }

  template<class T>
  using BaseType = std::decay_t<decltype(std::declval<T>()[0])>;

  template<class T, class Func>
  using BaseTypeFn = std::decay_t<decltype(std::declval<Func>()(std::declval<T>()[0]))>;

  template<class T, class IndexType>
    requires(!std::integral<IndexType>)
  static auto to_simd(T&& base, const IndexType& idx)
  {
    return load_rvalue<BaseType<T>>(base, idx);
  }

  template<class T, class IndexType, class Func>
    requires(!std::integral<IndexType>)
  static auto to_simd(T&& base, const IndexType& idx, Func&& subelement)
  {
    return load_rvalue<BaseTypeFn<T, Func>>(base, idx, subelement);
  }
};

/**
 * This function mocks a globally overloaded operator[]. It can be used instead of the SIMD_ACCESS macro,
 * if no named subobject is accessed.
 */
template<class T, class IndexType>
inline decltype(auto) sa(T&& base, const IndexType& index)
{
  return LValueSeparator<std::is_lvalue_reference_v<decltype(base[0])>>::to_simd(base, index);
}

template<has_to_simd T>
inline auto to_simd(const T& value)
{
  return value.to_simd();
}

template<class T> requires(!has_to_simd<T>)
inline auto to_simd(T&& value)
{
  return value;
}

} //namespace simd_access

/**
 * This macro defines a simd access to a variable of the form `base[index] subobject` (subobject is optional).
 * TODO: If global operator[] overloading becomes possible, then a decomposition of `base[index]` isn't required
 * anymore. Direct accesses (`base[index]`) and accesses to sub-arrays (`base[index][subindex]`) can be directly
 * written then.
 * TODO: If operator.() overloading becomes possible too, then this macro becomes obsolete, since all expressions can
 * be directly written.
 */
#define SIMD_ACCESS(base, index, ...) \
  simd_access::LValueSeparator<std::is_lvalue_reference_v<decltype((base[0] __VA_ARGS__))>>:: \
    to_simd(base, index __VA_OPT__(, [&](auto&& e) -> decltype((e __VA_ARGS__)) { return e __VA_ARGS__; }))

#define SIMD_ACCESS_V(...) simd_access::to_simd(SIMD_ACCESS(__VA_ARGS__))

#endif //SIMD_ACCESS_MAIN
