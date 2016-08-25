# C++ SIMD
Practical header only, oop way to use simd extensions. It won't be the absolute
fastest (although best attempts are made at inlining) way to do everything, but
if you want to take advantage of simd without looking up crazy intrinsic names
and keeping track of registers and cache, it's a step forward.

This is currently unfinished, and mostly just supports SSE. Happy to accept
pull requests for NEON and others.

Example
```c++
Vect128f value(0.3);
value += Vect128f(2.3);
value /= Vect128i(1, 2, 3, 4);
assert_eq(value, Vect128f((0.3 + 2.3) / 1.0,
                          (0.3 + 2.3) / 2.0,
                          (0.3 + 2.3) / 3.0,
                          (0.3 + 2.3) / 4.0));
```

This example is contrived, but essentially it lets you use normal person
operators on simd vectors.

There are storing, loading, conversion, and most math operations defined.
Should be easy to implement anything yourself (pull request please!).

Check the source or unit tests for more info.
