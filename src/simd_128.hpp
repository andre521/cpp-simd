#pragma once

#include "simd.hpp"
#include <cstdint>
#include <algorithm>

namespace sight {
class Vect128f;  // forward declared

/**
 * @brief 128 bit vector of int32
 */
class Vect128i {
  private:
    __m128i val;

  public:
    /**
     * @brief Empty vector
     */
    inline Vect128i() {}

    /**
     * @brief Fill vector with i
     *
     * @param i value to set every entry to
     */
    inline explicit Vect128i(int32_t i) {
        val = _mm_set1_epi32(i);
    }

    /**
     * @brief Convert native __m128i to abstract Vect128i
     *
     * @param v vector to use
     */
    inline Vect128i(__m128i v) : val(v) {}  // NOLINT(runtime/explicit)

    /**
     * @brief Fill vector with values
     *
     * @param i0, i1, i2, i3 values to use
     */
    inline Vect128i(int32_t i0, int32_t i1, int32_t i2, int32_t i3) {
        val = _mm_setr_epi32(i0, i1, i2, i3);
    }

    /**
     * @brief Access value directly
     *
     * Be aware that this method does not do bounds checking, and that it is
     * probably the most inefficient way to do anything - only use for
     * debugging
     *
     * @param idx index in vector
     */
    inline int32_t operator[](unsigned int idx) const {
        int32_t array[4];
        storeu(array);
        return array[idx];
    }

    /**
     * @brief Loads vector values from an arbitrary point
     *
     * It is preferable to use an aligned pointer, as it can use the aligned
     * load operator, which performs operations much faster in most CPUs
     *
     * @param p loads 128 bits starting at p
     */
    static inline Vect128i loadu(const int32_t* p) {
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    }

    /**
     * @brief Loads vector values from an aligned pointer in memory
     *
     * This operation is preferable because the compiler can guarantee an
     * aligned load operation, which is faster on most CPUs
     *
     * @param p loads 128 bits starting at p
     */
    static inline Vect128i load(const int32_t* p) {
        return _mm_load_si128(reinterpret_cast<const __m128i*>(p));
    }

    /**
     * @brief Inserts the vector in a point in memory
     *
     * It is preferable to use an aligned pointer, as it can use the aligned
     * store operator, which performs operations much faster in most CPUs
     *
     * @param p stores 128 bits starting at p
     */
    inline void storeu(int32_t* p) const {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(p), val);
    }

    /**
     * @brief Inserts this vector in a aligned point in memory
     *
     * This operation is preferable because the compiler can guarantee an
     * aligned store operation, which is faster on most CPUs
     *
     * @param p stores 128 bits starting at p
     */
    inline void store(int32_t* p) const {
        _mm_store_si128(reinterpret_cast<__m128i*>(p), val);
    }

    /**
     * @brief Sets vector values to native __m128i
     *
     * @param v vector to use
     */
    inline void operator=(__m128i v) {
        val = v;
    }

    /**
     * @brief Converts to a native __m128i
     */
    inline operator __m128i() const {
        return val;
    }

    /**
     * @brief Converts to a floating point representation
     */
    inline operator Vect128f() const;

    /**
     * @brief Performs NOT (r[i] = ~this[i])
     */
    inline Vect128i operator~() const {
        return operator^(Vect128i(0xFFFFFFFF));
    }

    /**
     * @brief Adds vectors together (r[i] = this[i] + v[i])
     *
     * @param v vector to add
     */
    inline Vect128i operator+(const Vect128i& v) const {
        return _mm_add_epi32(val, v);
    }

    /**
     * @brief Subtracts vector (r[i] = this[i] - v[i])
     *
     * @param v vector to subtract
     */
    inline Vect128i operator-(const Vect128i& v) const {
        return _mm_sub_epi32(val, v);
    }

    /**
     * @brief Multiplies vectors together (r[i] = this[i] * v[i])
     *
     * @param v vector to multiply
     */
    inline Vect128i operator*(const Vect128i& v) const {
        #ifdef HAVE_SSE41
        return _mm_mullo_epi32(val, v);
        #else
        // https://stackoverflow.com/questions/10500766/sse-multiplication-of-4-32-bit-integers
        auto t1 = _mm_mul_epu32(val, v);
        auto t2 = _mm_mul_epu32(_mm_srli_si128(val, 4), _mm_srli_si128(v, 4));
        return _mm_unpacklo_epi32(
                   _mm_shuffle_epi32(t1, _MM_SHUFFLE(0, 0, 2, 0)),
                   _mm_shuffle_epi32(t2, _MM_SHUFFLE(0, 0, 2, 0)));
        #endif
    }

    /**
     * @brief Performs AND comparison (r[i] = this[i] & v[i])
     *
     * @param v vector to perform comparison with
     */
    inline Vect128i operator&(const Vect128i& v) const {
        return _mm_and_si128(val, v);
    }

    /**
     * @brief Performs OR comparison (r[i] = this[i] | v[i])
     *
     * @param v vector to perform comparison with
     */
    inline Vect128i operator|(const Vect128i& v) const {
        return _mm_or_si128(val, v);
    }

    /**
     * @brief Performs XOR comparison (r[i] = this[i] ^ v[i])
     *
     * @param v vector to perform comparison with
     */
    inline Vect128i operator^(const Vect128i& v) const {
        return _mm_xor_si128(val, v);
    }

    /**
     * @brief Performs less-than comparison (r[i] = this[i] < v[i])
     *
     * @param v vector to compare
     */
    inline Vect128i operator<(const Vect128i& v) const {
        return _mm_cmplt_epi32(val, v);
    }

    /**
     * @brief Performs less-than-or-equal-to comparison (r[i] = this[i] <= v[i])
     *
     * @param v vector to compare
     */
    inline Vect128i operator<=(const Vect128i& v) const {
        return ~(operator>(v));
    }

    /**
     * @brief Performs larger-than comparison (r[i] = this[i] > v[i])
     *
     * @param v vector to compare
     */
    inline Vect128i operator>(const Vect128i& v) const {
        return _mm_cmpgt_epi32(val, v);
    }

    /**
     * @brief Performs greater-than-or-equal-to comparison
     * (r[i] = this[i] >= v[i])
     *
     * @param v vector to compare
     */
    inline Vect128i operator>=(const Vect128i& v) const {
        return ~(operator<(v));
    }

    /**
     * @brief Performs equal-to comparison (r[i] = this[i] == v[i])
     *
     * @param v vector to compare
     */
    inline Vect128i operator==(const Vect128i& v) const {
        return _mm_cmpeq_epi32(val, v);
    }

    /**
     * @brief Performs not-equal-to comparison (r[i] = this[i] != v[i])
     *
     * @param v vector to compare
     */
    inline Vect128i operator!=(const Vect128i& v) const {
        return ~(operator==(v));
    }

    /**
     * @brief Add vector (this[i] = this[i] + v[i])
     *
     * @param v vector to add
     */
    inline void operator+=(const Vect128i& v) {
        val = operator+(v);
    }

    /**
     * @brief Subtracts vector (this[i] = this[i] - v[i])
     *
     * @param v vector to subtract
     */
    inline void operator-=(const Vect128i& v) {
        val = operator-(v);
    }

    /**
     * @brief Multiples vector (this[i] = this[i] * v[i])
     *
     * @param v vector to multiply
     */
    inline void operator*=(const Vect128i& v) {
        val = operator*(v);
    }

    /**
     * @brief AND operation with vector (this[i] = this[i] & v[i])
     *
     * @param v vector to use
     */
    inline void operator&=(const Vect128i& v) {
        val = operator&(v);
    }

    /**
     * @brief OR operation with vector (this[i] = this[i] | v[i])
     *
     * @param v vector to use
     */
    inline void operator|=(const Vect128i& v) {
        val = operator|(v);
    }

    /**
     * @brief XOR operation with vector (this[i] = this[i] ^ v[i])
     *
     * @param v vector to use
     */
    inline void operator^=(const Vect128i& v) {
        val = operator^(v);
    }
};

/**
 * @brief 128 bit vector of float32
 */
class Vect128f {
  private:
    __m128 val;

  public:
    /**
     * @brief Empty vector
     */
    inline Vect128f() {}

    /**
     * @brief Fill vector with i
     *
     * @param i value to set every entry to
     */
    inline explicit Vect128f(float i) {
        val = _mm_set1_ps(i);
    }

    /**
     * @brief Convert native __m128 to abstract Vect128f
     *
     * @param v vector to use
     */
    inline Vect128f(__m128 v) : val(v) {}  // NOLINT(runtime/explicit)

    /**
     * @brief Fill vector with values
     *
     * @param i0, i1, i2, i3 values to use
     */
    inline Vect128f(float i0, float i1, float i2, float i3) {
        val = _mm_setr_ps(i0, i1, i2, i3);
    }

    /**
     * @brief Access value directly
     *
     * Be aware that this method does not do bounds checking, and that it is
     * probably the most inefficient way to do anything - only use for
     * debugging
     *
     * @param idx index in vector
     */
    inline float operator[](unsigned int idx) const {
        float array[4];
        storeu(array);
        return array[idx];
    }

    /**
     * @brief Loads vector values from an arbitrary point
     *
     * It is preferable to use an aligned pointer, as it can use the aligned
     * load operator, which performs operations much faster in most CPUs
     *
     * @param p loads 128 bits starting at p
     */
    static inline Vect128f loadu(const float* p) {
        return _mm_loadu_ps(p);
    }

    /**
     * @brief Loads vector values from an aligned pointer in memory
     *
     * This operation is preferable because the compiler can guarantee an
     * aligned load operation, which is faster on most CPUs
     *
     * @param p loads 128 bits starting at p
     */
    static inline Vect128f load(const float* p) {
        return _mm_load_ps(p);
    }

    /**
     * @brief Inserts the vector in a point in memory
     *
     * It is preferable to use an aligned pointer, as it can use the aligned
     * store operator, which performs operations much faster in most CPUs
     *
     * @param p stores 128 bits starting at p
     */
    inline void storeu(float* p) const {
        _mm_storeu_ps(p, val);
    }

    /**
     * @brief Inserts this vector in a aligned point in memory
     *
     * This operation is preferable because the compiler can guarantee an
     * aligned store operation, which is faster on most CPUs
     *
     * @param p stores 128 bits starting at p
     */
    inline void store(float* p) const {
        _mm_store_ps(p, val);
    }

    /**
     * @brief Sets vector values to native __m128
     *
     * @param v vector to use
     */
    inline void operator=(__m128 v) {
        val = v;
    }

    /**
     * @brief Converts to a native __m128
     */
    inline operator __m128() const {
        return val;
    }

    /**
     * @brief Converts to a integer representation (flooring)
     */
    inline Vect128i to_int() const;

    /**
     * @brief Performs NOT (r[i] = ~this[i])
     */
    inline Vect128f operator~() const {
        return operator^(Vect128f(0xFFFFFFFF));
    }

    /**
     * @brief Adds vectors together (r[i] = this[i] + v[i])
     *
     * @param v vector to add
     */
    inline Vect128f operator+(const Vect128f& v) const {
        return _mm_add_ps(val, v);
    }

    /**
     * @brief Subtracts vector (r[i] = this[i] - v[i])
     *
     * @param v vector to subtract
     */
    inline Vect128f operator-(const Vect128f& v) const {
        return _mm_sub_ps(val, v);
    }

    /**
     * @brief Multiplies vectors together (r[i] = this[i] * v[i])
     *
     * @param v vector to multiply
     */
    inline Vect128f operator*(const Vect128f& v) const {
        return _mm_mul_ps(val, v);
    }

    /**
     * @brief Divides vectors (r[i] = this[i] / v[i])
     *
     * @param v vector to divide by
     */
    inline Vect128f operator/(const Vect128f& v) const {
        return _mm_div_ps(val, v);
    }

    /**
     * @brief Performs AND comparison (r[i] = this[i] & v[i])
     *
     * @param v vector to perform comparison with
     */
    inline Vect128f operator&(const Vect128f& v) const {
        return _mm_and_ps(val, v);
    }

    /**
     * @brief Performs OR comparison (r[i] = this[i] | v[i])
     *
     * @param v vector to perform comparison with
     */
    inline Vect128f operator|(const Vect128f& v) const {
        return _mm_or_ps(val, v);
    }

    /**
     * @brief Performs XOR comparison (r[i] = this[i] ^ v[i])
     *
     * @param v vector to perform comparison with
     */
    inline Vect128f operator^(const Vect128f& v) const {
        return _mm_xor_ps(val, v);
    }

    /**
     * @brief Performs less-than comparison (r[i] = this[i] < v[i])
     *
     * @param v vector to compare
     */
    inline Vect128f operator<(const Vect128f& v) const {
        return _mm_cmplt_ps(val, v);
    }

    /**
     * @brief Performs less-than-or-equal-to comparison (r[i] = this[i] <= v[i])
     *
     * @param v vector to compare
     */
    inline Vect128f operator<=(const Vect128f& v) const {
        return _mm_cmpngt_ps(val, v);
    }

    /**
     * @brief Performs larger-than comparison (r[i] = this[i] > v[i])
     *
     * @param v vector to compare
     */
    inline Vect128f operator>(const Vect128f& v) const {
        return _mm_cmpgt_ps(val, v);
    }

    /**
     * @brief Performs greater-than-or-equal-to comparison
     * (r[i] = this[i] >= v[i])
     *
     * @param v vector to compare
     */
    inline Vect128f operator>=(const Vect128f& v) const {
        return _mm_cmpnlt_ps(val, v);
    }

    /**
     * @brief Performs equal-to comparison (r[i] = this[i] == v[i])
     *
     * @param v vector to compare
     */
    inline Vect128f operator==(const Vect128f& v) const {
        return _mm_cmpeq_ps(val, v);
    }

    /**
     * @brief Performs not-equal-to comparison (r[i] = this[i] != v[i])
     *
     * @param v vector to compare
     */
    inline Vect128f operator!=(const Vect128f& v) const {
        return _mm_cmpneq_ps(val, v);
    }

    /**
     * @brief Add vector (this[i] = this[i] + v[i])
     *
     * @param v vector to add
     */
    inline void operator+=(const Vect128f& v) {
        val = operator+(v);
    }

    /**
     * @brief Subtracts vector (this[i] = this[i] - v[i])
     *
     * @param v vector to subtract
     */
    inline void operator-=(const Vect128f& v) {
        val = operator-(v);
    }

    /**
     * @brief Multiples vector (this[i] = this[i] * v[i])
     *
     * @param v vector to multiply
     */
    inline void operator*=(const Vect128f& v) {
        val = operator*(v);
    }

    /**
     * @brief AND operation with vector (this[i] = this[i] & v[i])
     *
     * @param v vector to use
     */
    inline void operator&=(const Vect128f& v) {
        val = operator&(v);
    }

    /**
     * @brief OR operation with vector (this[i] = this[i] | v[i])
     *
     * @param v vector to use
     */
    inline void operator|=(const Vect128f& v) {
        val = operator|(v);
    }

    /**
     * @brief XOR operation with vector (this[i] = this[i] ^ v[i])
     *
     * @param v vector to use
     */
    inline void operator^=(const Vect128f& v) {
        val = operator^(v);
    }
};

Vect128i::operator Vect128f() const {
    return _mm_cvtepi32_ps(val);
}

Vect128i Vect128f::to_int() const {
    return _mm_cvttps_epi32(val);
}

/**
 * @brief Returns lowest of each value
 *
 * @param v first vector
 * @param v2 second vector
 * @return minimum of v[i] and v2[i]
 */
inline Vect128i lowest(const Vect128i& v, const Vect128i& v2) {
    #ifdef HAVE_SSE41
    return _mm_min_epi32(v, v2);
    #else
    return Vect128i(std::min(v[0], v2[0]),
                    std::min(v[1], v2[1]),
                    std::min(v[2], v2[2]),
                    std::min(v[3], v2[3]));
    #endif
}

/**
 * @brief Returns lowest of each value
 *
 * @param v first vector
 * @param v2 second vector
 * @return minimum of v[i] and v2[i]
 */
inline Vect128f lowest(const Vect128f& v, const Vect128f& v2) {
    return _mm_min_ps(v, v2);
}

/**
 * @brief Returns highest of each value
 *
 * @param v first vector
 * @param v2 second vector
 * @return maximum of v[i] and v2[i]
 */
inline Vect128i highest(const Vect128i& v, const Vect128i& v2) {
    #ifdef HAVE_SSE41
    return _mm_max_epi32(v, v2);
    #else
    return Vect128i(std::max(v[0], v2[0]),
                    std::max(v[1], v2[1]),
                    std::max(v[2], v2[2]),
                    std::max(v[3], v2[3]));
    #endif
}

/**
 * @brief Returns highest of each value
 *
 * @param v first vector
 * @param v2 second vector
 * @return maximum of v[i] and v2[i]
 */
inline Vect128f highest(const Vect128f& v, const Vect128f& v2) {
    return _mm_max_ps(v, v2);
}

/**
 * @brief Rounds each value to closest integer
 *
 * @param v vector of values to round
 * @return rounded values of v[i]
 */
inline Vect128i round(const Vect128f& v) {
    return (v + Vect128f(0.5)).to_int();
}

/**
 * @brief The reciprocal square root of values in a vector
 *
 * @param v starting values
 * @return 1 / sqrt(v[i])
 */
inline Vect128f rsqrt(const Vect128f& v) {
    return _mm_rsqrt_ps(v);
}

/**
 * @brief The reciprocal of values in a vector
 *
 * @param v starting values
 * @return 1 / v[i]
 */
inline Vect128f reciprocal(const Vect128f& v) {
    return _mm_rcp_ps(v);
}

/**
 * @brief The square root of values in a vector
 *
 * This method isn't guaranteed to be accurate
 *
 * @param v starting values
 * @return the square root of v[i]
 */
inline Vect128f sqrt(const Vect128f& v) {
    return reciprocal(rsqrt(v));
}

}  // namespace sight
