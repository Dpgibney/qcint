/*
 * The Rys roots of a SIMD chunk of points at once (AVX and AVX-512 builds): the macros of rys_roots.c's chunk
 * functions and the one-root case, inlined by rys_roots.c (CINTrys_roots_chunk) and g1e_grids.c.
 */

#if !defined(HAVE_DEFINED_RYS_ROOTS_CHUNK_H) && (defined(__AVX512F__) || defined(__AVX__))
#define HAVE_DEFINED_RYS_ROOTS_CHUNK_H
#define HAVE_RYS_ROOTS_CHUNK

#include <math.h>
#include "simd.h"

#define CHUNK_PIE4      0.78539816339744827900  // PIE4 of rys_roots.c
#if defined(__AVX512F__)
#define CHUNK_LANES     0xff
#define CHUNK_CMP(x, edge)  _mm512_cmp_pd_mask(x, MM_SET1(edge), _CMP_GT_OQ)
// (x > edge) ? a : b
#define CHUNK_SELECT(x, edge, a, b) _mm512_mask_blend_pd(_mm512_cmp_pd_mask(x, MM_SET1(edge), _CMP_GT_OQ), b, a)
#define CHUNK_CMPGE(x, edge)  _mm512_cmp_pd_mask(x, MM_SET1(edge), _CMP_GE_OQ)
// (x >= edge) ? a : b
#define CHUNK_SELECT_GE(x, edge, a, b) _mm512_mask_blend_pd(_mm512_cmp_pd_mask(x, MM_SET1(edge), _CMP_GE_OQ), b, a)
#else
#define CHUNK_LANES     0xf
#define CHUNK_CMP(x, edge)  _mm256_movemask_pd(_mm256_cmp_pd(x, MM_SET1(edge), _CMP_GT_OQ))
#define CHUNK_SELECT(x, edge, a, b) _mm256_blendv_pd(b, a, _mm256_cmp_pd(x, MM_SET1(edge), _CMP_GT_OQ))
#define CHUNK_CMPGE(x, edge)  _mm256_movemask_pd(_mm256_cmp_pd(x, MM_SET1(edge), _CMP_GE_OQ))
#define CHUNK_SELECT_GE(x, edge, a, b) _mm256_blendv_pd(b, a, _mm256_cmp_pd(x, MM_SET1(edge), _CMP_GE_OQ))
#endif

/*
 * CINTrys_roots(1, x) at the lanes of X: rys_root1, whose branch x > 33 is the large-x form. The fit of each
 * interval of x is evaluated on the whole chunk when a lane is in it, and selected by x; exp(-x) of the lanes in
 * one loop, which the compiler vectorizes where the C library has vector exp. *pu is the root u, or with t2 = 1
 * the t^2 = u/(u+1) of the grids' integrals, formed directly (F1/w instead of u = F1/(w-F1), 0.5/x instead of
 * u = 0.5/(x-0.5)). *pw is the weight. Returns 0, with nothing done, if a lane is at x <= 3e-7 (the small-x form).
 */
static inline int CINTrys_root1_chunk(__MD X, __MD *pu, __MD *pw, const int t2)
{
        const __MD rhalf = MM_SET1(.5);
        const __MD rpie4 = MM_SET1(CHUNK_PIE4);
        // 0.5/x = 0.5/PIE4 * (PIE4/x), the square of the weight's factor
        const __MD rc = MM_SET1(.5 / CHUNK_PIE4);
        __MD r, ww;
        int large = CHUNK_CMP(X, 33.);
        if (large == CHUNK_LANES) {
                ww = MM_SQRT(rpie4 / X);
                r = t2 ? rc * ww * ww : rhalf / (X - rhalf);
        } else {
                if (CHUNK_CMP(X, 3.e-7) != CHUNK_LANES) {
                        return 0;
                }
                int gt1 = CHUNK_CMP(X, 1.);
                int gt3 = CHUNK_CMP(X, 3.);
                int gt5 = CHUNK_CMP(X, 5.);
                int gt10 = CHUNK_CMP(X, 10.);
                int gt15 = CHUNK_CMP(X, 15.);
                ALIGNMM double xs[SIMDD];
                ALIGNMM double es[SIMDD];
                MM_STORE(xs, X);
                int l;
                for (l = 0; l < SIMDD; l++) {
                        es[l] = exp(-xs[l]);
                }
                __MD E = MM_LOAD(es);
                __MD Y, F1, F;
                F1 = MM_SET1(0.);
                if (~gt1 & CHUNK_LANES) {
                        F1 = ((((((((-8.36313918003957E-08*X+1.21222603512827E-06 )*X-
                                    1.15662609053481E-05 )*X+9.25197374512647E-05 )*X-
                                  6.40994113129432E-04 )*X+3.78787044215009E-03 )*X-
                                1.85185172458485E-02 )*X+7.14285713298222E-02 )*X-
                              1.99999999997023E-01 )*X+3.33333333333318E-01;
                }
                if (gt1 & ~gt3) {
                        Y = X-2.0E+00;
                        F = ((((((((((-1.61702782425558E-10*Y+1.96215250865776E-09 )*Y-
                                      2.14234468198419E-08 )*Y+2.17216556336318E-07 )*Y-
                                    1.98850171329371E-06 )*Y+1.62429321438911E-05 )*Y-
                                  1.16740298039895E-04 )*Y+7.24888732052332E-04 )*Y-
                                3.79490003707156E-03 )*Y+1.61723488664661E-02 )*Y-
                              5.29428148329736E-02 )*Y+1.15702180856167E-01;
                        F1 = CHUNK_SELECT(X, 1., F, F1);
                }
                if (gt3 & ~gt5) {
                        Y = X-4.0E+00;
                        F = ((((((((((-2.62453564772299E-11*Y+3.24031041623823E-10 )*Y-
                                      3.614965656163E-09)*Y+3.760256799971E-08)*Y-
                                    3.553558319675E-07)*Y+3.022556449731E-06)*Y-
                                  2.290098979647E-05)*Y+1.526537461148E-04)*Y-
                                8.81947375894379E-04 )*Y+4.33207949514611E-03 )*Y-
                              1.75257821619926E-02 )*Y+5.28406320615584E-02;
                        F1 = CHUNK_SELECT(X, 3., F, F1);
                }
                if (gt5 & ~gt10) {
                        Y = 1./X;
                        F = ((((((( 4.6897511375022E-01*Y-6.9955602298985E-01)*Y +
                                   5.3689283271887E-01)*Y-3.2883030418398E-01)*Y +
                                 2.4645596956002E-01)*Y-4.9984072848436E-01)*Y -
                               3.1501078774085E-06)* E + MM_SQRT(rpie4/X) - E)*Y;
                        F *= .5;
                        F1 = CHUNK_SELECT(X, 5., F, F1);
                }
                if (gt10 & ~gt15) {
                        Y = 1./X;
                        F = ((((-1.8784686463512E-01*Y+2.2991849164985E-01)*Y -
                                4.9893752514047E-01)*Y-2.1916512131607E-05)* E
                                + MM_SQRT(rpie4/X) - E)*Y;
                        F *= .5;
                        F1 = CHUNK_SELECT(X, 10., F, F1);
                }
                if (gt15 & ~large) {
                        Y = 1./X;
                        F = ((( 1.9623264149430E-01*Y-4.9695241464490E-01)*Y -
                               6.0156581186481E-05)* E + MM_SQRT(rpie4/X) - E)*Y;
                        F *= .5;
                        F1 = CHUNK_SELECT(X, 15., F, F1);
                }
                ww = 2. * X * F1 + E;
                r = t2 ? F1 / ww : F1 / (ww - F1);
                if (large) {
                        __MD s = MM_SQRT(rpie4 / X);
                        r = CHUNK_SELECT(X, 33., t2 ? rc * s * s : rhalf / (X - rhalf), r);
                        ww = CHUNK_SELECT(X, 33., s, ww);
                }
        }
        *pu = r;
        *pw = ww;
        return 1;
}

#endif
