#include <gtest/gtest.h>

#include "simd.hpp"
#include <cmath>

using namespace sight;

template <typename T, typename B>
void checkEqual(T p, B p1, size_t size) {
    for (int x = 0; x < size; x++) {
        ASSERT_EQ(p[x], p1[x]);
    }
}

TEST(simd, aligned_ptr) {
    AlignedStorage<float, 128> f(256);
    ASSERT_EQ(0, ((uintptr_t) (float*) f) % 128);
    AlignedStorage<long long, 17> l(256);
    ASSERT_EQ(0, ((uintptr_t) (long long*) l) % 17);
    AlignedStorage<char, 4096> c(256);
    ASSERT_EQ(0, ((uintptr_t) (char*) c) % 4096);

    f[1] = 1.1;
    ASSERT_FLOAT_EQ(1.1, *(f + 1));
    l[1] = 11;
    ASSERT_EQ(11, *(l + 1));
    c[1] = 11;
    ASSERT_EQ(11, *(c + 1));
    f.clear();
    ASSERT_FLOAT_EQ(0, *(f + 1));
    l.clear();
    ASSERT_FLOAT_EQ(0, *(l + 1));
    c.clear();
    ASSERT_FLOAT_EQ(0, *(c + 1));
}

TEST(simd, vect128i_construction) {
    {
        Vect128i vect;
        auto vect2 = Vect128i();
        auto vect3 = vect;
        vect3 = vect2;
    }

    {
        int x[4] = {0, 1, 2, 3};
        Vect128i i = Vect128i::loadu(x);
        checkEqual(i, x, 4);
    }

    {
        AlignedStorage<int32_t, 16> p(4);
        p[0] = 0, p[1] = 1, p[2] = 2, p[3] = 3;
        Vect128i i = Vect128i::load(p);
        checkEqual(i, p, 4);
    }

    {
        int q[4];
        Vect128i i(0, 1, 2, 3);
        i.storeu(q);
        checkEqual(q, i, 4);
    }

    {
        AlignedStorage<int32_t, 16> p(4);
        Vect128i i(0, 1, 2, 3);
        i.store(p);
        checkEqual(i, p, 4);
    }

    {
        Vect128i i(0, 1, 2, 3);
        Vect128i d; d = i;
        checkEqual(d, i, 4);
    }

    {
        Vect128i i(1);
        int r[4] = {1, 1, 1, 1};
        checkEqual(i, r, 4);
    }

    {
        Vect128i p(0, 1, 2, 3);
        int s[4] = {0, 1, 2, 3};
        checkEqual(p, s, 4);
    }

    {
        Vect128i p(0, 1, 2, 3);
        __m128i m = p;
        Vect128i pd = m;
        int s[4] = {0, 1, 2, 3};
        checkEqual(pd, s, 4);
    }
}

TEST(simd, vect128i_data) {
    {
        Vect128i v(-1, 1, 2147483647, -2147483648);
        int x[4] = {-1, 1, 2147483647, -2147483648};
        checkEqual(v, x, 4);
    }

    {
        Vect128i v;
        int x[4] = {-1, 1, 2147483647, -2147483648};
        v = Vect128i::loadu(x);
        checkEqual(v, x, 4);
    }
}

TEST(simd, vect128i_operators) {
    {
        Vect128i v(0xFF00FF00);
        Vect128i s(0x00FF00FF);
        checkEqual(v, ~s, 4);
        checkEqual(s, ~v, 4);
    }

    {
        Vect128i i(0, -1, 1, 2147483647);
        Vect128i s(1,  1, 1,          1);
        Vect128i r(1,  0, 2, 2147483648);
        checkEqual(r, i + s, 4);
        checkEqual(r, s + i, 4);

        i += s;
        checkEqual(r, i, 4);
    }

    {
        Vect128i i( 0, -1, 1, 2147483648);
        Vect128i s( 1,  1, 1,          1);
        Vect128i r(-1, -2, 0, 2147483647);
        checkEqual(r, i - s, 4);

        i -= s;
        checkEqual(r, i, 4);
    }

    {
        Vect128i i(0, -1, 1, 2147483648);
        Vect128i s(1,  1, 1,          1);
        Vect128i r(0, -1, 1, 2147483648);
        checkEqual(r, i * s, 4);
        checkEqual(r, s * i, 4);

        i *= s;
        checkEqual(r, i, 4);
    }

    {
        Vect128i i(0xF0F10);
        Vect128i s(0xF001F);
        Vect128i r(0xF0F10 & 0xF001F);
        checkEqual(r, i & s, 4);

        i &= s;
        checkEqual(r, i, 4);
    }

    {
        Vect128i i(0xF0F10);
        Vect128i s(0xF001F);
        Vect128i r(0xF0F10 | 0xF001F);
        checkEqual(r, i | s, 4);

        i |= s;
        checkEqual(r, i, 4);
    }

    {
        Vect128i i(0xF0F10);
        Vect128i s(0xF001F);
        Vect128i r(0xF0F10 ^ 0xF001F);
        checkEqual(r, i ^ s, 4);

        i ^= s;
        checkEqual(r, i, 4);
    }

    {
        Vect128i i(0,  1, -1, 2147483647);
        Vect128i s(0,  0,  0,          0);
        Vect128i r(0, -1,  0,         -1);
        checkEqual(r, i > s, 4);
    }

    {
        Vect128i i(0, 1, -1, 2147483647);
        Vect128i s(0, 0,  0,          0);
        Vect128i r(0, 0, -1,          0);
        checkEqual(r, i < s, 4);
    }

    {
        Vect128i i( 0,  1, -1, 2147483647);
        Vect128i s( 0,  0,  0,          0);
        Vect128i r(-1, -1,  0,         -1);
        checkEqual(r, i >= s, 4);
    }

    {
        Vect128i i( 0, 1, -1, 2147483647);
        Vect128i s( 0, 0,  0,          0);
        Vect128i r(-1, 0, -1,          0);
        checkEqual(r, i <= s, 4);
    }

    {
        Vect128i i( 0, 1, -1, 2147483647);
        Vect128i s( 0, 0,  0,          0);
        Vect128i r(-1, 0,  0,          0);
        checkEqual(r, i == s, 4);
    }

    {
        Vect128i i(0,  1, -1, 2147483647);
        Vect128i s(0,  0,  0,          0);
        Vect128i r(0, -1, -1,         -1);
        checkEqual(r, i != s, 4);
    }
}

TEST(simd, vect128f_construction) {
    {
        Vect128f vect;
        auto vect2 = Vect128f();
        auto vect3 = vect;
        vect3 = vect2;
    }

    {
        Vect128f vect(23);
        Vect128i vect2 = vect.to_int();
        checkEqual(vect, vect2, 4);
    }

    {
        Vect128i vect(23);
        Vect128f vect2 = vect;
        checkEqual(vect, vect2, 4);
    }

    {
        float x[4] = {0, 0.1, 1, 2};
        Vect128f i = Vect128f::loadu(x);
        checkEqual(i, x, 4);
    }

    {
        AlignedStorage<float, 16> p(4);
        p[0] = 0, p[1] = 0.1, p[2] = 1, p[3] = 2;
        Vect128f i = Vect128f::load(p);
        checkEqual(i, p, 4);
    }

    {
        float q[4];
        Vect128f i(0, 0.1, 1, 2);
        i.storeu(q);
        checkEqual(q, i, 4);
    }

    {
        AlignedStorage<float, 16> p(4);
        Vect128f i(0, 0.1, 1, 2);
        i.store(p);
        checkEqual(i, p, 4);
    }

    {
        Vect128f i(0, 0.1, 1, 2);
        Vect128f d; d = i;
        checkEqual(d, i, 4);
    }

    {
        Vect128f i(1);
        float r[4] = {1, 1, 1, 1};
        checkEqual(i, r, 4);
    }

    {
        Vect128f p(0, 0.1, 1, 2);
        float s[4] = {0, 0.1, 1, 2};
        checkEqual(p, s, 4);
    }

    {
        Vect128f p(0, 0.1, 1, 2);
        __m128 m = p;
        Vect128f pd = m;
        float s[4] = {0, 0.1, 1, 2};
        checkEqual(pd, s, 4);
    }
}

TEST(simd, vect128f_data) {
    {
        Vect128f v(-1, 1, 3.14, -3.4e+29);
        float x[4] = {-1, 1, 3.14, -3.4e+29};
        checkEqual(v, x, 4);
    }

    {
        Vect128f v;
        float x[4] = {-1, 1, 3.14, -3.4e+29};
        v = Vect128f::loadu(x);
        checkEqual(v, x, 4);
    }
}

TEST(simd, vect128f_operators) {
    {
        Vect128f v(1.203);
        auto v2 = ~v;
        auto r = v2 == v;
        ASSERT_FALSE(std::isnan(r[0]));
        ASSERT_FALSE(std::isnan(r[1]));
        ASSERT_FALSE(std::isnan(r[2]));
        ASSERT_FALSE(std::isnan(r[3]));
        auto v3 = ~v2;
        auto r2 = v3 == v;
        ASSERT_TRUE(std::isnan(r2[0]));
        ASSERT_TRUE(std::isnan(r2[1]));
        ASSERT_TRUE(std::isnan(r2[2]));
        ASSERT_TRUE(std::isnan(r2[3]));
    }

    {
        Vect128f i(0, -1, 1,  1);
        Vect128f s(1,  1, 1, -2);
        Vect128f r(1,  0, 2, -1);
        checkEqual(r, i + s, 4);
        checkEqual(r, s + i, 4);

        i += s;
        checkEqual(r, i, 4);
    }

    {
        Vect128f i( 0, -1, 1,  1);
        Vect128f s( 1,  1, 1, -2);
        Vect128f r(-1, -2, 0,  3);
        checkEqual(r, i - s, 4);

        i -= s;
        checkEqual(r, i, 4);
    }

    {
        Vect128f i(0, -1, 1,  1);
        Vect128f s(1,  1, 1, -2);
        Vect128f r(0, -1, 1, -2);
        checkEqual(r, i * s, 4);
        checkEqual(r, s * i, 4);

        i *= s;
        checkEqual(r, i, 4);
    }

    {
        Vect128f i(0.1);
        Vect128f s(0.0);
        Vect128f r(0.0);
        checkEqual(r, i & s, 4);

        i &= s;
        checkEqual(r, i, 4);

        Vect128f i2(0.1);
        Vect128f s2(0.1);
        Vect128f r2(0.1);
        checkEqual(r2, i2 & s2, 4);

        i2 &= s2;
        checkEqual(r2, i2, 4);
    }

    {
        Vect128f i(0.1);
        Vect128f s(0.0);
        Vect128f r(0.1);
        checkEqual(r, i | s, 4);

        i |= s;
        checkEqual(r, i, 4);

        Vect128f i2(0.1);
        Vect128f s2(0.1);
        Vect128f r2(0.1);
        checkEqual(r2, i2 | s2, 4);

        i2 |= s2;
        checkEqual(r2, i2, 4);

        Vect128f i3(0.0);
        Vect128f s3(0.0);
        Vect128f r3(0.0);
        checkEqual(r3, i3 | s3, 4);

        i3 |= s3;
        checkEqual(r3, i3, 4);
    }

    {
        Vect128f i(0.1);
        Vect128f s(0.0);
        Vect128f r(0.1);
        checkEqual(r, i ^ s, 4);

        i ^= s;
        checkEqual(r, i, 4);

        Vect128f i2(0.1);
        Vect128f s2(0.1);
        Vect128f r2(0.0);
        checkEqual(r2, i2 ^ s2, 4);

        i2 ^= s2;
        checkEqual(r2, i2, 4);

        Vect128f i3(0.0);
        Vect128f s3(0.0);
        Vect128f r3(0.0);
        checkEqual(r3, i3 ^ s3, 4);

        i3 ^= s3;
        checkEqual(r3, i3, 4);
    }

    {
        Vect128f i(0,  1, -1, 3.4e+29);
        Vect128f s(0,  0,  0,       1);
        auto r = i > s;
        ASSERT_FALSE(std::isnan(r[0]));
        ASSERT_TRUE(std::isnan(r[1]));
        ASSERT_FALSE(std::isnan(r[2]));
        ASSERT_TRUE(std::isnan(r[3]));
    }

    {
        Vect128f i(0, 1, -1, 3.4e+29);
        Vect128f s(0, 0,  0,       1);
        auto r = i < s;
        ASSERT_FALSE(std::isnan(r[0]));
        ASSERT_FALSE(std::isnan(r[1]));
        ASSERT_TRUE(std::isnan(r[2]));
        ASSERT_FALSE(std::isnan(r[3]));
    }

    {
        Vect128f i(0,  1, -1, 3.4e+29);
        Vect128f s(0,  0,  0,       1);
        auto r = i >= s;
        ASSERT_TRUE(std::isnan(r[0]));
        ASSERT_TRUE(std::isnan(r[1]));
        ASSERT_FALSE(std::isnan(r[2]));
        ASSERT_TRUE(std::isnan(r[3]));
    }

    {
        Vect128f i(0, 1, -1, 3.4e+29);
        Vect128f s(0, 0,  0,       0);
        auto r = i <= s;
        ASSERT_TRUE(std::isnan(r[0]));
        ASSERT_FALSE(std::isnan(r[1]));
        ASSERT_TRUE(std::isnan(r[2]));
        ASSERT_FALSE(std::isnan(r[3]));
    }

    {
        Vect128f i(0, 1, -1, 2147483647);
        Vect128f s(0, 0,  0,          0);
        auto r = i == s;
        ASSERT_TRUE(std::isnan(r[0]));
        ASSERT_FALSE(std::isnan(r[1]));
        ASSERT_FALSE(std::isnan(r[2]));
        ASSERT_FALSE(std::isnan(r[3]));
    }

    {
        Vect128f i(0,  1, -1, 2147483647);
        Vect128f s(0,  0,  0,          0);
        auto r = i != s;
        ASSERT_FALSE(std::isnan(r[0]));
        ASSERT_TRUE(std::isnan(r[1]));
        ASSERT_TRUE(std::isnan(r[2]));
        ASSERT_TRUE(std::isnan(r[3]));
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
