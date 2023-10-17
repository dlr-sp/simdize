#include <iostream>

#include <experimental/simd>

namespace stdx = std::experimental;

template<size_t N> struct simd_index_array;
template<size_t N> struct simd_index;
template<class T, class SimdIndexType, size_t ElementSize> struct simd_access_expr;
template<class T> concept is_class = std::is_class_v<T>;
template<class T> concept is_not_class = !std::is_class_v<T>;

template<size_t Pitch, size_t N, class T>
auto gather_pitched(T* base_addr)
{
  return stdx::fixed_size_simd<std::remove_const_t<T>, N>([&](int i)
    {
      return *reinterpret_cast<T*>(reinterpret_cast<const char*>(base_addr) + Pitch * i);
    });
}

template<size_t Pitch, size_t N, class T>
auto scatter_pitched(T* base_addr, const stdx::fixed_size_simd<T, N>& value)
{
  for (size_t i = 0; i < N; ++i)
  {
    *reinterpret_cast<T*>(reinterpret_cast<char*>(base_addr) + Pitch * i) = value[i];
  }
}


template<typename T, size_t N, size_t ElementSize> struct dot_overload;

template<is_not_class T, size_t N, size_t ElementSize>
struct dot_overload<T, N, ElementSize> {};

template<is_class T, size_t N, size_t ElementSize>
struct dot_overload<T, N, ElementSize>
{
  template<auto T::*Member>
  auto dot()
  {
    T& base = *static_cast<simd_access_expr<T, simd_index<N>, ElementSize>*>(this)->base_;
    return simd_access_expr<std::remove_reference_t<decltype(base.*Member)>, simd_index<N>, ElementSize>{{}, &(base.*Member)};
  }
};


template<class T, size_t N, size_t ElementSize>
struct simd_access_expr<T, simd_index<N>, ElementSize> : dot_overload<T, N, ElementSize>
{
  T* base_;

  template<class SIMD_SOURCE>
  void operator=(const SIMD_SOURCE& x)
  {
    static_assert(N == x.size());
    if constexpr (sizeof(T) == ElementSize)
    {
      x.copy_to(base_, stdx::element_aligned);
    }
    else
    {
      return scatter_pitched<ElementSize, N>(base_, x);
    }
  }

  operator stdx::fixed_size_simd<std::remove_const_t<T>, N>() const
  {
    return to_simd();
  }

  auto to_simd() const
  {
    if constexpr (sizeof(T) == ElementSize)
    {
      return stdx::fixed_size_simd<std::remove_const_t<T>, N>(base_, stdx::element_aligned);
    }
    else
    {
      return gather_pitched<ElementSize, N>(base_);
    }
  }

  auto operator[](int i)
  {
    return simd_access_expr<std::remove_pointer_t<decltype((*base_) + i)>, simd_index<N>, ElementSize>{{}, (*base_) + i};
  }
};

template<class T, size_t N, size_t ElementSize>
struct simd_access_expr<T, simd_index_array<N>, ElementSize>
{
  T* data_;
  const simd_index_array<N>& index_;

  template<class SIMD_SOURCE>
  void operator=(const SIMD_SOURCE& x)
  {
    static_assert(N == x.size());
    for (size_t i = 0; i < N; ++i)
    {
      data_[index_.index_[i]] = x[i];
    }
  }

  operator stdx::fixed_size_simd<std::remove_const_t<T>, N>() const
  {
    return stdx::fixed_size_simd<std::remove_const_t<T>, N>([&](int i) { return data_[index_.index_[i]]; });
  }

  auto operator[](int i)
  {
    return simd_access_expr<T, simd_index_array<N>, ElementSize>{data_ + i, index_};
  }
};

template<class T, class U, class V, size_t ElementSize>
inline auto operator/(const V& v1, const simd_access_expr<T, U, ElementSize>& v2)
{
  return v1 / v2.to_simd();
}



template<size_t N>
struct simd_index_array
{
  static constexpr size_t size = N;
  size_t index_[size];

  template<class T>
  auto operator[](T* data) const
  {
    return simd_access_expr<T, simd_index_array<N>, sizeof(T)>{data, *this};
  }
};

template<size_t N>
struct simd_index
{
  static constexpr size_t size = N;
  size_t index_;

  template<class T>
  auto operator[](T* data) const
  {
    return simd_access_expr<T, simd_index<N>, sizeof(T)>{ {}, data + index_};
  }
};



template<class T, size_t N>
auto Subscript(T* data, simd_index<N> idx)
{
  return simd_access_expr<T, simd_index<N>, sizeof(T)>{ {}, data, idx};
}

template<class T>
auto& Subscript(T* data, size_t idx)
{
  return data[idx];
}


template<class T, class IndexType>
void SimdAgnosticFunction(T* destination, const T* source, const IndexType& i)
{
  i[destination] = 1.0 / i[source];
}

template<size_t size, class FUNC>
void SimdLoop(FUNC&& f, size_t array_size)
{
  simd_index<size> simd_i{0};
  for (; simd_i.index_ < array_size - size + 1; simd_i.index_ += size)
  {
    f(simd_i);
  }
  for (size_t i = simd_i.index_; i < array_size; ++i)
  {
    f(i);
  }
}

void DoRealWork()
{
  const size_t array_size = 101;
  double source[array_size], destination[array_size];
  for (int i = 0; i < array_size; ++i) { source[i] = i + 1.0; }

  SimdLoop<stdx::native_simd<double>::size()>([&](auto i) { SimdAgnosticFunction(destination, source, i); }, array_size);

Test:
  for (size_t i = 0; i < array_size; ++i)
  {
    if (destination[i] != 1.0 / (i + 1.0))
    {
      std::cout << "error at " << i << ", expected " << 1.0 / (i + 1.0) << ", got " << destination[i] << "\n";
    }
  }
}

template<class T, class IndexType>
void SimdAgnosticArrayFunction(T* destination, const T* source, const IndexType& i)
{
  for (int j = 0; j < 3; ++j)
  {
    i[destination][j] = 1.0 / i[source][j];
  }
}

void DoRealArrayWork()
{
  const size_t array_size = 101;
  double source[array_size][3], destination[array_size][3];
  for (int i = 0; i < array_size; ++i) for (int j = 0; j < 3; ++j) { source[i][j] = i + j + 1.0; }

  SimdLoop<stdx::native_simd<double>::size()>([&](auto i) { SimdAgnosticArrayFunction(destination, source, i); }, array_size);

Test:
  for (size_t i = 0; i < array_size; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      if (destination[i][j] != 1.0 / (i + j + 1.0))
      {
        std::cout << "error at " << i << ", expected " << 1.0 / (i + 1.0) << ", got " << destination[i][j] << "\n";
      }
    }
  }
}


struct Vector
{
  double x, y;

  template<auto Vector::*Member>
  auto& dot()
  {
    return this->*Member;
  }

  template<auto Vector::*Member>
  const auto& dot() const
  {
    return this->*Member;
  }
};

template<class T, class IndexType>
void SimdAgnosticVectorFunction(T* destination, const T* source, const IndexType& i)
{
  i[destination].template dot<&T::x>() = 1.0 / i[source].template dot<&T::x>();
  i[destination].template dot<&T::y>() = 1.0 / i[source].template dot<&T::y>();
}

void DoRealVectorWork()
{
  const size_t array_size = 101;
  Vector source[array_size], destination[array_size];
  for (int i = 0; i < array_size; ++i) { source[i].x = i + 1.0; source[i].y = i + 2.0; }

  SimdLoop<stdx::native_simd<double>::size()>([&](auto i) { SimdAgnosticVectorFunction(destination, source, i); }, array_size);

Test:
  for (size_t i = 0; i < array_size; ++i)
  {
    if (destination[i].x != 1.0 / (i + 1.0))
    {
      std::cout << "error at " << i << ", expected " << 1.0 / (i + 1.0) << ", got " << destination[i].x << "\n";
    }
    if (destination[i].y != 1.0 / (i + 2.0))
    {
      std::cout << "error at " << i << ", expected " << 1.0 / (i + 2.0) << ", got " << destination[i].y << "\n";
    }
  }
}


int main()
{
  DoRealWork();
  DoRealArrayWork();
  DoRealVectorWork();
  std::cout << "Finished\n";
}



#if (0)

template<class T, size_t N>
auto SubscriptLoad(const T* data, simd_index<N> idx)
{
  return stdx::simd<T, stdx::simd_abi::fixed_size<N>>([](int i) { return data[idx.index_ + i]; });
}

template<class T, size_t N>
auto SubscriptLoad(const T* data, const simd_index_array<N>& idx)
{
  return stdx::simd<T, stdx::simd_abi::fixed_size<N>>([](int i) { return data[idx.index_[i]]; });
}


template<class Func, size_t SimdLen>
inline auto Load(SimdIndex<SimdLen> idx, Func&& f)
{
  using SimdType = SimpleSimd<decltype(f(size_t()))>;
  static_assert(SimdLen == SimdType::SimdLen, "mismatching simd types");
  SimdType t;
  for (size_t i = 0; i < SimdType::SimdLen; ++i) t.data_[i] = f(idx.data_+i);
  return t;
}

template<class Func, size_t SimdLen>
inline auto Load(const SimdIndexArray<SimdLen>& idx, Func&& f)
{
  using SimdType = SimpleSimd<decltype(f(size_t()))>;
  static_assert(SimdLen == SimdType::SimdLen, "mismatching simd types");
  SimdType t;
  for (size_t i = 0; i < SimdLen; ++i) t.data_[i] = f(idx.data_[i]);
  return t;
}

template<class Func>
inline auto Load(LocalIndexT idx, Func&& f)
{
  return f(idx);
}

template<class Func, class ValueType>
inline void Store(LocalIndexT idx, ValueType& v, Func&& f)
{
  f(idx, v);
}

template<class Func, class ValueType, size_t SimdLen>
inline void Store(SimdIndex<SimdLen> idx, ValueType& v, Func&& f)
{
  using SimdType = ValueType;
  for (size_t i = 0; i < SimdType::SimdLen; ++i) f(idx.data_+i, v.data_[i]);
}

template<class Func, class ValueType, size_t SimdLen>
inline void Store(const SimdIndexArray<SimdLen>& idx, ValueType& v, Func&& f)
{
  using SimdType = ValueType;
  static_assert(SimdLen == SimdType::SimdLen, "mismatching simd types");
  for (size_t i = 0; i < SimdType::SimdLen; ++i) f(idx.data_[i], v.data_[i]);
}


template<class ScalarT, class IdxType>
auto get_base_addr(ScalarT* base_addr, IdxType i) { return base_addr + i; }

template<class ScalarT, size_t SimdLen>
auto get_base_addr(ScalarT* base_addr, flis::SimdIndex<SimdLen> i) { return base_addr + i.data_; }

template<class ScalarT, size_t SimdLen>
auto get_base_addr(ScalarT* base_addr, flis::SimdIndexArray<SimdLen>) { return base_addr; }


template<size_t ElementSize, class ScalarT, size_t SimdLen>
void simd_store_dispatch(const ScalarT* base_addr, flis::SimdIndex<SimdLen>, const flis::SimpleSimd<ScalarT>& x)
{
  if constexpr(ElementSize == sizeof(*base_addr))
  {
    for (size_t i = 0; i < SimdLen; ++i)
    {
      base_addr[i] = x.data_[i];
    }
  }
  else
  {
    for (size_t i = 0; i < SimdLen; ++i)
    {
      *base_addr = x.data_[i];
      base_addr = reinterpret_cast<const ScalarT*>(reinterpret_cast<const char*>(base_addr) + ElementSize);
    }
  }
}

template<size_t ElementSize, class ScalarT, size_t SimdLen>
void simd_store_dispatch(const ScalarT* base_addr, const flis::SimdIndexArray<SimdLen>& array,
  const flis::SimpleSimd<ScalarT>& x)
{
  if constexpr(ElementSize == sizeof(*base_addr))
  {
    for (size_t i = 0; i < SimdLen; ++i)
    {
      base_addr[array.data_[i]] = x.data_[i];
    }
  }
  else
  {
    auto base_addr_byte = reinterpret_cast<const char*>(base_addr);
    for (size_t i = 0; i < SimdLen; ++i)
    {
      *reinterpret_cast<const ScalarT*>(base_addr_byte + array.data_[i] * sizeof(ElementSize)) = x.data_[i];
    }
  }
}

template<class ScalarT, class SimdIndexType, size_t ElementSize>
struct simd_access_expr
{
  ScalarT* base_addr;
  const SimdIndexType& index;

  void operator=(const flis::SimpleSimd<ScalarT>& x)
  {
    simd_store_dispatch<ElementSize>(base_addr, index, x);
  }

  template<class T>
  auto operator*(const T& other) const
  {
    return simd_load_dispatch(*this) * other;
  }
};

template<class T, class ScalarT, class SimdIndexType, size_t ElementSize>
auto operator/(const T& other, const simd_access_expr<ScalarT, SimdIndexType, ElementSize>& access)
{
  return other / simd_load_dispatch(access);
}

template<size_t ElementSize, class ScalarT, class IdxType>
auto simd_access(ScalarT* base_addr, IdxType) { return *base_addr; }

template<size_t ElementSize, class ScalarT, size_t SimdLen>
auto simd_access(ScalarT* base_addr, const flis::SimdIndex<SimdLen>& i)
{
  return simd_access_expr<ScalarT, flis::SimdIndex<SimdLen>, ElementSize>{base_addr, i};
}

template<size_t ElementSize, class ScalarT, size_t SimdLen>
auto simd_access(ScalarT* base_addr, const flis::SimdIndexArray<SimdLen>& array)
{
  return simd_access_expr<ScalarT, flis::SimdIndex<SimdLen>, ElementSize>{base_addr, array};
}

template<size_t ElementSize, class ScalarT, size_t SimdLen>
auto simd_load_dispatch(const simd_access_expr<ScalarT, flis::SimdIndex<SimdLen>, ElementSize>& expr)
{
  if constexpr(ElementSize == sizeof(*expr.base_addr))
  {
    return flis::SimpleSimd<decltype(*expr.base_addr)>(expr.base_addr);
  }
  else
  {
    return flis::SimpleSimd<decltype(*expr.base_addr)>(expr.base_addr, std::integral_constant<size_t, ElementSize>());
  }
}

template<size_t ElementSize, class ScalarT, size_t SimdLen>
auto simd_load_dispatch(const simd_access_expr<ScalarT, flis::SimdIndexArray<SimdLen>, ElementSize>& expr)
{
  flis::SimpleSimd<ScalarT> result;
  if constexpr(ElementSize == sizeof(*expr.base_addr))
  {
    for (size_t i = 0; i < SimdLen; ++i)
    {
      result.data_[i] = expr.base_addr[expr.index.data_[i]];
    }
  }
  else
  {
    auto base_addr_byte = reinterpret_cast<const char*>(expr.base_addr);
    for (size_t i = 0; i < SimdLen; ++i)
    {
      result.data_[i] = *reinterpret_cast<const ScalarT*>(base_addr_byte + expr.index.data_[i] * sizeof(ElementSize));
    }
  }
  return result;
}

#define SIMDIZE(base, index, member) \
  simd_access<sizeof(base[index])>(&((*get_base_addr(base, index)) member ), index)


template<typename ScalarType, typename Index>
void ComputeValue(ScalarType* dest, const ScalarType* source1, const ScalarType* source2, Index i)
{
  SIMDIZE(dest, i) = SIMDIZE(source1, i) * (ScalarType(2.0) / SIMDIZE(source2, i));
}

// original code
template<typename value_type>
struct Properties
{
  value_type length_, area_, volume_;
};

template<typename value_type, size_t number_of_elements>
struct Elements
{
  Properties<value_type> properties_[number_of_elements];
};

template<typename value_type, size_t number_of_elements>
using Gradients = std::array<value_type, number_of_elements>;

template<typename value_type, size_t number_of_elements>
struct OriginalFunctor
{
  using elements_type = Elements<value_type, number_of_elements>;
  using gradients_type = Gradients<value_type, number_of_elements>;

  OriginalFunctor(const elements_type* elements, gradients_type* gradients, const value_type* correction)
    : elements_(elements), gradients_(gradients), correction_(correction)
  { }

  void operator()(int elemIndex) const
  {
    gradients_[elemIndex][0] *= 1.0 / elements_[elemIndex].properties_[0].volume;
    for (size_t i = 1; i < number_of_elements; ++i)
    {
      gradients_[elemIndex][0] += correction_[elemIndex];
    }
  }

  const elements_type* elements_;
  const value_type* correction_;
  gradients_type* gradients_;
};


template<typename value_type, size_t number_of_elements>
struct SimdizeFunctor
{
  using elements_type = Elements<value_type, number_of_elements>;
  using gradients_type = Gradients<value_type, number_of_elements>;

  SimdizeFunctor(const elements_type* elements, gradients_type* gradients, const value_type* correction)
    : elements_(elements), gradients_(gradients), correction_(correction)
  { }

  // envisioned form:
  template<typename index_type>
  void operator()(const index_type& elemIndex) const
  {
    gradients_[elemIndex][0] *= 1.0 / elements_[elemIndex].properties_[0].volume;
    for (size_t i = 1; i < number_of_elements; ++i)
    {
      gradients_[elemIndex][0] += correction_[elemIndex];
    }
  }

  const elements_type* elements_;
  const value_type* correction_;
  gradients_type* gradients_;

  static constexpr auto simd_len = simdize::SimpleSimd<value_type>::simd_len;
};



template<typename value_type>
struct OriginalFunctor
{
  OriginalFunctor(const Elements &elems, ElementGradientArrayType &elemStateGradient)
    : _elems(elems.GetMetricData().GetPointerConst()), _elemStateGradient(elemStateGradient.GetPointer())
  { }

  template<class index_type>
  void operator()(const index_type elemIndex) const
  {
      _elemStateGradient[elemIndex][0] *= 1.0 / _elems[elemIndex].points[0].volume;
  }

  static constexpr auto SimdLen = simdize::SimpleSimd<value_type>::SimdLen;

protected:
  const typename Elements::Element *const _elems;
  typename ElementGradientArrayType::ValueT *_elemStateGradient;
};



template<class Fun>
void Loop(size_t range, Fun&& f)
{
  for (size_t i = 0; i < range; ++i)
  {
    f(i);
  }
}


template<class Fun>
void SimdLoop(size_t range, Fun&& f)
{
  for (size_t i = 0; i < range; ++i)
  {
    f(i);
  }
}

#endif