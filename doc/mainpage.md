# simd_access - Thinking One Step Further.

simd_access is meant as a supplement to `std::experimental::simd` (in short `stdx::simd`).
The aim is to syntactically equalize the loading and saving of simd variables to the
loading and saving of usual variables.
Then loop bodies written for scalar variables can be used without changes for simd variables.


