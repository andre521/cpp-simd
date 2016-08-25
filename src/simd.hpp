#pragma once

#ifdef __SSE2__
    #define HAVE_SSE2
    #define HAVE_SSE
#endif
#ifdef __SSSE3__
    #define HAVE_SSE3
    #define HAVE_SSE
#endif
#ifdef __SSE4_1__
    #define HAVE_SSE41
    #define HAVE_SSE
#endif
#ifdef __SSE4_2__
    #define HAVE_SSE42
    #define HAVE_SSE
#endif
#ifdef __AVX__
    #define HAVE_AVX
#endif
#ifdef __AVX2__
    #define HAVE_AVX2
#endif

#if defined(HAVE_SSE) \
    || defined(HAVE_AVX) \
    || defined(HAVE_AVX2)
    #include <x86intrin.h>
#else
    #error "SSE/AVX is required for compiling"
#endif

#include <memory>

namespace sight {

/**
 * @brief Smart pointer that aligns to boundary for fast load & store
 *
 * @param T type of underlying data
 * @param Align alignment in memory
 */
template <typename T, int Align>
class AlignedStorage {
  public:
    /// Length of allocated data
    const size_t length;

    /**
     * @brief Allocates data
     *
     * @param length amount of data to allocate (how many T's)
     */
    explicit AlignedStorage(size_t length)
        : length(length),
          raw(static_cast<T*>(malloc((length * sizeof(T)) + Align)), free),
          raw_val(reinterpret_cast<uintptr_t>(raw.get())),
          aligned(reinterpret_cast<T*>(raw_val + (Align - (raw_val % Align)))) {
    }

    /// Sets all data to 0
    inline void clear() const {
        memset(aligned, 0, length * sizeof(T));
    }

    /// Casts to a non-const pointer
    inline operator T*() {
        return aligned;
    }

    /// Casts to a const pointer
    inline operator const T*() const {
        return aligned;
    }

    /**
     * @brief Returns an offset pointer
     *
     * @param i offset from main pointer
     * @return pointer of data + i
     */
    inline T* operator+(int i) {
        return static_cast<T*>(aligned + i);
    }

    /**
     * @brief Returns an offset const pointer
     *
     * @param i offset from main pointer
     * @return pointer of data + i
     */
    inline const T* operator+(int i) const {
        return static_cast<const T*>(aligned + i);
    }

    /**
     * @brief Returns an offset pointer
     *
     * @param i offset from main pointer
     * @return pointer of data - i
     */
    inline T* operator-(int i) {
        return aligned - i;
    }

    /**
     * @brief Returns an offset const pointer
     *
     * @param i offset from main pointer
     * @return pointer of data - i
     */
    inline const T* operator-(int i) const {
        return aligned - i;
    }

    /**
     * @brief Returns data at an index
     *
     * @param i index of data
     * @return writable reference to data
     */
    inline T& operator[](int i) {
        return aligned[i];
    }

    /**
     * @brief Returns data at an index
     *
     * @param i index of data
     * @return data found at index
     */
    inline const T& operator[](int i) const {
        return aligned[i];
    }

    /**
     * @brief Returns data at an index, with a bounds check
     *
     * @param i index of data
     * @return writable reference to data
     */
    inline T& at(int i) {
        if (i < 0 || i >= length) {
            throw std::out_of_range("outside of aligned storage boundary");
        }
        return aligned[i];
    }

    /**
     * @brief Returns data at an index, with a bounds check
     *
     * @param i index of data
     * @return data found at index
     */
    inline const T& at(int i) const {
        if (i < 0 || i >= length) {
            throw std::out_of_range("outside of aligned storage boundary");
        }
        return aligned[i];
    }

  private:
    const std::shared_ptr<T> raw;
    const uintptr_t raw_val;
    T* aligned;
};

/**
 * @brief Checks if a pointer is properly aligned to a boundary
 *
 * @param Align alignment of pointer
 * @param T type of data to be used
 * @param ptr pointer to data you want to be aligned
 * @return if ptr is aligned to Align (ie. ptr % Align == 0)
 */
template <int Align, typename T>
inline bool isAligned(const T* ptr) {
    return reinterpret_cast<uintptr_t>(ptr) % Align == 0;
}

/**
 * @brief Checks if there is enough room in length for M more elements
 *
 * @param M elements in a "leap"
 * @param index current index inside of length
 * @param length total length
 * @return index + M <= length
 */
template <int M>
inline bool room(int index, int length) {
    return index + M <= length;
}

}  // namespace sight

// SIMD implementations
#include "simd_128.hpp"
