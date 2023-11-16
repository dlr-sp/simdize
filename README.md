# SIMD Classes - Thinking one Step Further.

**simd_access** is meant as a supplement to `std::experimental::simd` (in short `stdx::simd`).
The aim is to syntactically equalize the loading and saving of simd variables to the loading and saving of
usual variables.
Then loop bodies written for scalar variables can be used without changes for simd variables.

## State of the Art

There are a lot of simd libraries.
They all have in common, that they provide some kind of a simd type and overload the usual operators for that type.
Thus, once you have simd variables, you can handle them as usual scalar variables.
However, they also all have in common (as far as I am aware), that obtaining and storing simd variables
is very different from obtaining and storing scalar variables.
In the function
```c++
void times2(int size, double* dest, const double* source)
{
  for (int i = 0; i < size; ++i)
  {
    dest[i] = source[i] * 2;
  }
}
```
you can't just use the line `dest[i] = source[i] * 2;`, if you want to explicitly simdize this function.
If you want to preserve the look and feel, you need a compiler extension (e.g. `omp simd`).
On the other hand, compiler extensions are often not capable to automatically vectorize complex codes
in the intended way.

## The Core Idea

A typical loop that is suitable for vectorization has three general steps:
```c++
for (int i = 0; i < size; ++i)
{
  // (1) load data associated with i to variables
  // (2) compute results
  // (3) store results to locations associated with i
}
```
In a scalar loop the three steps can and often will be intertwined.
By using a simd library this becomes difficult, the steps are often separated.
Then the second step can already be written in the same way for scalar and vectorized variables with the help of a
simd library like `stdx::simd`.
C++ achieves this by deducing the matching operators according to the types of the variables created in the first step.

The core idea of the `simd_access` library is now to push the boundary of the type deduction out of the three steps.
Not only the computation, but also load and store operations are deduced.
The deduction is done according to the type of the loop index.
You don't use an integral type as loop index anymore, but a `simd_index`.
Then step (1) and (3) both can deduce the matching operations according to the type of the index.
In addition, it will once again be possible to intertwine all three steps for vectorized code in the way
you are used to.
A simplified version of `times2` (with improper residual loop handling) then could look as follows:
```c++
void times2(int size, double* dest, const double* source)
{
  auto simd_size = stdx::native_simd<double>::size()
  for (simd_index<simd_size> i = 0; i < size; i += simd_size)
  {
    dest[i] = source[i] * 2;
  }
}
```
In this code the expressions `source[i]` and `dest[i]` load and store simd variables according to `simd_size` encoded
as a template paramter in `simd_index`. As usual, the multiplication is then also recognized as a simd operation.
Unfortunately, the current C++ language has some shortcomings, so that the above code is not possible yet.
However, the `simd_access` library provides a macro `SIMD_ACCESS` in addition to a `simd_index`
to achieve a similar effect.
The library also serves as a basis for discussion for some language extensions, with which the above code ultimately
becomes legal C++.

## The `simd_access` Library

All functions and types of the library reside in the `simd_access` namespace (in short `sa`).
The library contains four general components:
1. A `SIMD_ACCESS` macro, which can be used to access variables as simd or scalar variables (depending on the context).
1. An `sa::loop` function, which iterates over a given range of indices. Residual iterations are handled properly.
1. An outline for a proposal to enable a global `operator[]` overload.
1. An outline for a proposal to enable a member access overload (`operator.`).

### The `SIMD_ACCESS` Macro

### The `sa::loop` Function

### A globally overloadable subscription operator (`operator[]`)

### An overloadable member access operator (`operator.`)

### Build Requirements

The lib is header-only. The tests need cmake and a c++20 compliant compiler.
