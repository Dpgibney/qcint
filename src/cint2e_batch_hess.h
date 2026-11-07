/*
 * Qcint is a general GTO integral library for computational chemistry
 * Copyright (C) 2014- Qiming Sun <osirpt.sun@gmail.com>
 *
 * This file is part of Qcint.
 *
 * Qcint is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * The two-electron integrals of nuclear Hessians, several shell quartets of one class at once, the
 * quartets in the SIMD lanes as cint2e_batch.h does for int2e and int2e_ip1:
 *   int2e_ipip1_sph   (nabla nabla i, j|k, l)
 *   int2e_ipvip1_sph  (nabla i, nabla j|k, l)
 *   int2e_ip1ip2_sph  (nabla i, j|nabla k, l)
 * with 9 components each. As int2e_sph's scalar path (autocode/hess.c, g2e_simd1.c), the 2D
 * integrals go one higher on each centre that is differentiated (two on i for ipip1), from the same
 * base centres; then g1 = nabla_J g0 (nabla_K for ip1ip2, nabla_I for ipip1), g2 = nabla_I g0 and
 * g3 = nabla_I g1, with nabla f(n) = n f(n-1) - 2 a f(n+1). Component 3a+b holds nabla of
 * direction a on i and of direction b on j (k for ip1ip2, i for ipip1).
 *
 * The _dot kernels contract the integrals with ng (at most HESS_MAXG) arrays gam of each quartet, in
 * int2e_sph's layout: out[(q*ng+n)*9+c] = sum_f int[q][c][f] gam[(q*ng+n)*nf+f]. gam goes to the
 * cartesian functions by the transpose of the cart -> sph step, once per quartet and array, and is
 * contracted with the cartesian integrals: no component goes to spherical. (Adding each root sum
 * to the 9 x ng sums at once instead was slower for s and p shells: the sums do not stay in
 * registers.)
 */
#include "cint2e_batch.h"

#define HESS_IPIP1      2
#define HESS_IPVIP1     3
#define HESS_IP1IP2     4

#define HESS_MAXR       8       // Rys roots of the second derivatives of (ff|ff)
#define HESS_MAXG       2       // arrays of a _dot kernel
// workspace in bvec: the 2D integrals [roots][3][(li+1+dI)(lj+1+dJ)(lk+1+dK)(ll+1)] (at most 400
// per root and direction), g0..g3 at the output's powers [4][roots][3][(li+1)(lj+1)(lk+1)(ll+1)],
// the cartesian integrals [9][...], and two arrays of the cart -> sph transform of one component
#define HESS_NE_MAX     (HESS_MAXR * 3 * 400)
#define HESS_NO_MAX     (HESS_MAXR * 3 * 256)
#define HESS_NC_MAX     (9 * 10000)
#define HESS_NT1_MAX    7000
#define HESS_NT2_MAX    4900
#define HESS_CACHE_BVEC (HESS_NE_MAX + 4 * HESS_NO_MAX + HESS_NC_MAX + HESS_NT1_MAX + HESS_NT2_MAX)

BATCH_HIDDEN void CINTbatch2e_ipip1_l0(double *out, int nq, CINTBatch2e *b, double omega, bvec *cache, int lj, int lk, int ll);
BATCH_HIDDEN void CINTbatch2e_ipip1_l1(double *out, int nq, CINTBatch2e *b, double omega, bvec *cache, int lj, int lk, int ll);
BATCH_HIDDEN void CINTbatch2e_ipip1_l2(double *out, int nq, CINTBatch2e *b, double omega, bvec *cache, int lj, int lk, int ll);
BATCH_HIDDEN void CINTbatch2e_ipip1_l3(double *out, int nq, CINTBatch2e *b, double omega, bvec *cache, int lj, int lk, int ll);
BATCH_HIDDEN void CINTbatch2e_ipvip1_l0(double *out, int nq, CINTBatch2e *b, double omega, bvec *cache, int lj, int lk, int ll);
BATCH_HIDDEN void CINTbatch2e_ipvip1_l1(double *out, int nq, CINTBatch2e *b, double omega, bvec *cache, int lj, int lk, int ll);
BATCH_HIDDEN void CINTbatch2e_ipvip1_l2(double *out, int nq, CINTBatch2e *b, double omega, bvec *cache, int lj, int lk, int ll);
BATCH_HIDDEN void CINTbatch2e_ipvip1_l3(double *out, int nq, CINTBatch2e *b, double omega, bvec *cache, int lj, int lk, int ll);
BATCH_HIDDEN void CINTbatch2e_ip1ip2_l0(double *out, int nq, CINTBatch2e *b, double omega, bvec *cache, int lj, int lk, int ll);
BATCH_HIDDEN void CINTbatch2e_ip1ip2_l1(double *out, int nq, CINTBatch2e *b, double omega, bvec *cache, int lj, int lk, int ll);
BATCH_HIDDEN void CINTbatch2e_ip1ip2_l2(double *out, int nq, CINTBatch2e *b, double omega, bvec *cache, int lj, int lk, int ll);
BATCH_HIDDEN void CINTbatch2e_ip1ip2_l3(double *out, int nq, CINTBatch2e *b, double omega, bvec *cache, int lj, int lk, int ll);
#define HESS_DECLARE_DOT(NAME, LI) \
BATCH_HIDDEN void CINTbatch2e_##NAME##_dot_l##LI(double *out, int nq, CINTBatch2e *b, double omega, \
                                                 bvec *cache, int lj, int lk, int ll, \
                                                 const double *gam, int ng);
HESS_DECLARE_DOT(ipip1, 0) HESS_DECLARE_DOT(ipip1, 1) HESS_DECLARE_DOT(ipip1, 2) HESS_DECLARE_DOT(ipip1, 3)
HESS_DECLARE_DOT(ipvip1, 0) HESS_DECLARE_DOT(ipvip1, 1) HESS_DECLARE_DOT(ipvip1, 2) HESS_DECLARE_DOT(ipvip1, 3)
HESS_DECLARE_DOT(ip1ip2, 0) HESS_DECLARE_DOT(ip1ip2, 1) HESS_DECLARE_DOT(ip1ip2, 2) HESS_DECLARE_DOT(ip1ip2, 3)

/*
 * The transpose of one cart -> sph pass (_batch_c2s): over the index of stride `inner` and extent ns
 * (-> nc), for every combination of `outer` outer and `inner` inner elements:
 * dst[o][c][x] = sum_s C2S[l][s][c] src[o][s][x]
 */
static inline __attribute__((always_inline))
void _batch_s2c(bvec *dst, const bvec *src, const int l, int outer, int inner)
{
        const int nc = (l + 1) * (l + 2) / 2, ns = 2 * l + 1;
        for (int o = 0; o < outer; o++)
        for (int c = 0; c < nc; c++) {
                bvec *d = dst + (o * nc + c) * inner;
                const bvec *x = src + o * ns * inner;
                for (int n = 0; n < inner; n++) d[n] = (bvec){0};
                for (int s = 0; s < ns; s++) {
                        const double f = BATCH_C2S[l][s][c];
                        if (f != 0) {
                                for (int n = 0; n < inner; n++) d[n] += f * x[s * inner + n];
                        }
                }
        }
}

// nabla of the 2D integrals e at index p along stride s, power n: n e(n-1) - 2a e(n+1), a2 = -2a,
// as CINTnabla1i_2e_simd1
static inline __attribute__((always_inline))
bvec _hess_nabla(const bvec *e, const int p, const int s, const int n, const bvec a2)
{
        if (n) {
                return n * e[p - s] + a2 * e[p + s];
        }
        return a2 * e[p + s];
}

static inline __attribute__((always_inline))
void CINTbatch2e_hess_t(double *out, int nq, CINTBatch2e *b, double omega, bvec *cache,
                        const int li, const int lj, const int lk, const int ll, const int kind,
                        const int dot, const double *gam, const int ng)
{
        const int dI = kind == HESS_IPIP1 ? 2 : 1;
        const int dJ = kind == HESS_IPVIP1;
        const int dK = kind == HESS_IP1IP2;
        const int lI = li + dI, lJ = lj + dJ, lK = lk + dK;
        const int nroots = (lI + lJ + lK + ll) / 2 + 1;
        const int nmax = lI + lJ, mmax = lK + ll;
        const int NI = lI + 1, NJ = lJ + 1, NK = lK + 1, NL = ll + 1;
        const int nE = NI * NJ * NK * NL;
        const int nO = (li + 1) * (lj + 1) * (lk + 1) * (ll + 1);
        const int nci = (li + 1) * (li + 2) / 2, ncj = (lj + 1) * (lj + 2) / 2;
        const int nck = (lk + 1) * (lk + 2) / 2, ncl = (ll + 1) * (ll + 2) / 2;
        const int nsi = 2 * li + 1, nsj = 2 * lj + 1, nsk = 2 * lk + 1, nsl = 2 * ll + 1;
        const int ncart = nci * ncj * nck * ncl;
        const int nf = nsi * nsj * nsk * nsl;
        bvec *g2d = cache;                      // [nroots][3][NI][NJ][NK][NL]
        bvec *G0 = g2d + nroots * 3 * nE;       // [nroots][3][li+1][lj+1][lk+1][ll+1]
        bvec *G1 = G0 + nroots * 3 * nO;
        bvec *G2 = G1 + nroots * 3 * nO;
        bvec *G3 = G2 + nroots * 3 * nO;
        bvec *gc = G3 + nroots * 3 * nO;        // [9][ncl][nck][ncj][nci]; the cartesian gam of _dot
        bvec *t1 = gc + 9 * ncart;
        bvec *t2 = t1 + HESS_NT1_MAX;
        const double common = M_PI * M_PI * M_PI * 2 / sqrt(M_PI)
                * _batch_fac_sp(li) * _batch_fac_sp(lj) * _batch_fac_sp(lk) * _batch_fac_sp(ll);
        // offsets of the powers of each cartesian function in G0..G3's [li+1][lj+1][lk+1][ll+1]
        int ox[4][10], oy[4][10], oz[4][10];
        const int ls[4] = {li, lj, lk, ll};
        const int stride[4] = {(lj + 1) * (lk + 1) * (ll + 1), (lk + 1) * (ll + 1), ll + 1, 1};
        for (int t = 0; t < 4; t++) {
                for (int c = 0; c < (ls[t] + 1) * (ls[t] + 2) / 2; c++) {
                        ox[t][c] = BATCH_CART[ls[t]][c][0] * stride[t];
                        oy[t][c] = BATCH_CART[ls[t]][c][1] * stride[t];
                        oz[t][c] = BATCH_CART[ls[t]][c][2] * stride[t];
                }
        }
        // the strides of i, j and k in g2d
        const int sI = NJ * NK * NL, sJ = NK * NL, sK = NL;

        // the base centres of the recurrences, as int2e_sph from the raised l: a = i if lI > lJ,
        // else j; c = k if lK > ll, else l
        const int ibase = lI > lJ, kbase = lK > ll;
        const int lb = ibase ? lJ : lI, ld = kbase ? ll : lK;
        const int ta = ibase ? 0 : 1, tc = kbase ? 2 : 3;
        bvec rab[3], rcd[3], rrij = {0}, rrkl = {0};
        for (int d = 0; d < 3; d++) {
                bvec rij = b->r[0][d] - b->r[1][d];
                bvec rkl = b->r[2][d] - b->r[3][d];
                rab[d] = ibase ? rij : -rij;
                rcd[d] = kbase ? rkl : -rkl;
                rrij += rij * rij;
                rrkl += rkl * rkl;
        }
        for (int n = 0; n < 9 * ncart; n++) gc[n] = (bvec){0};

        for (int pi = 0; pi < b->np[0]; pi++)
        for (int pj = 0; pj < b->np[1]; pj++)
        for (int pk = 0; pk < b->np[2]; pk++)
        for (int pl = 0; pl < b->np[3]; pl++) {
                const bvec ai = b->a[0][pi], aj = b->a[1][pj], ak = b->a[2][pk], al = b->a[3][pl];
                const bvec aij = ai + aj, akl = ak + al;
                const bvec cc = b->c[0][pi] * b->c[1][pj] * b->c[2][pk] * b->c[3][pl];
                bvec pa[3], qc[3], pq[3], rr = {0};
                for (int d = 0; d < 3; d++) {
                        bvec p = (ai * b->r[0][d] + aj * b->r[1][d]) / aij;
                        bvec q = (ak * b->r[2][d] + al * b->r[3][d]) / akl;
                        pa[d] = p - b->r[ta][d];
                        qc[d] = q - b->r[tc][d];
                        pq[d] = p - q;
                        rr += pq[d] * pq[d];
                }
                const bvec a1 = aij * akl;
                const bvec a0 = a1 / (aij + akl);
                bvec fac1;
                double u[HESS_MAXR][SIMDD] __attribute__((aligned(SIMDD * 8)));
                double w[HESS_MAXR][SIMDD] __attribute__((aligned(SIMDD * 8)));
                double xs[SIMDD] __attribute__((aligned(SIMDD * 8)));
                double theta[SIMDD] __attribute__((aligned(SIMDD * 8)));
                for (int lane = 0; lane < SIMDD; lane++) {
                        const double eij = exp(-ai[lane] * aj[lane] / aij[lane] * rrij[lane]);
                        const double ekl = exp(-ak[lane] * al[lane] / akl[lane] * rrkl[lane]);
                        const double a1l = a1[lane], a0l = a0[lane];
                        double f = sqrt(a0l / (a1l * a1l * a1l)) * common * cc[lane] * eij * ekl;
                        xs[lane] = a0l * rr[lane];
                        if (omega != 0) {
                                // long range, as int2e_sph
                                theta[lane] = omega * omega / (omega * omega + a0l);
                                f *= sqrt(theta[lane]);
                                xs[lane] *= theta[lane];
                        }
                        fac1[lane] = f;
                }
                _CINTrys_roots_batch(nroots, xs, &u[0][0], &w[0][0], SIMDD);
                if (omega != 0) {
                        for (int r = 0; r < nroots; r++)
                                for (int lane = 0; lane < SIMDD; lane++) {
                                        double ut = u[r][lane] * theta[lane];
                                        u[r][lane] = ut / (u[r][lane] + 1. - ut);
                                }
                }
                for (int r = 0; r < nroots; r++) {
                        bvec ur, wr;
                        memcpy(&ur, u[r], sizeof(bvec));
                        memcpy(&wr, w[r], sizeof(bvec));
                        const bvec u2 = a0 * ur;
                        const bvec tmp4 = .5 / (u2 * (aij + akl) + a1);
                        const bvec tmp5 = u2 * tmp4;
                        const bvec tmp1 = 2. * tmp5;
                        const bvec tmp2 = tmp1 * akl;
                        const bvec tmp3 = tmp1 * aij;
                        const bvec b00 = tmp5;
                        const bvec b10 = tmp5 + tmp4 * akl;
                        const bvec b01 = tmp5 + tmp4 * aij;
                        for (int d = 0; d < 3; d++) {
                                const bvec c00 = pa[d] - tmp2 * pq[d];
                                const bvec c0p = qc[d] + tmp3 * pq[d];
                                bvec g[2 * BATCH_MAXL + 3][2 * BATCH_MAXL + 2];  // [n][m]: (n 0| m 0)
                                g[0][0] = d == 2 ? wr * fac1 : ur - ur + 1;
                                if (nmax > 0) g[1][0] = c00 * g[0][0];
                                for (int n = 1; n < nmax; n++)
                                        g[n + 1][0] = c00 * g[n][0] + n * b10 * g[n - 1][0];
                                for (int m = 0; m < mmax; m++)
                                        for (int n = 0; n <= nmax; n++) {
                                                bvec v = c0p * g[n][m];
                                                if (m) v += m * b01 * g[n][m - 1];
                                                if (n) v += n * b00 * g[n - 1][m];
                                                g[n][m + 1] = v;
                                        }
                                // (a, b+1| = (a+1, b| + R_ab (a, b|, a the base centre of the bra
                                bvec h[BATCH_MAXL + 2][2 * BATCH_MAXL + 3][2 * BATCH_MAXL + 2];  // [b][a][m]
                                for (int n = 0; n <= nmax; n++)
                                        for (int m = 0; m <= mmax; m++) h[0][n][m] = g[n][m];
                                for (int nb = 1; nb <= lb; nb++)
                                        for (int na = 0; na <= nmax - nb; na++)
                                                for (int m = 0; m <= mmax; m++)
                                                        h[nb][na][m] = h[nb - 1][na + 1][m] + rab[d] * h[nb - 1][na][m];
                                bvec *E = g2d + (r * 3 + d) * nE;
                                for (int i = 0; i <= lI; i++)
                                for (int j = 0; j <= lJ; j++) {
                                        // |c, d+1) = |c+1, d) + R_cd |c, d), c the base centre of the ket
                                        const bvec *row = ibase ? h[j][i] : h[i][j];
                                        bvec kk[BATCH_MAXL + 1][2 * BATCH_MAXL + 2];  // [d][c]
                                        for (int m = 0; m <= mmax; m++) kk[0][m] = row[m];
                                        for (int nd = 1; nd <= ld; nd++)
                                                for (int nc = 0; nc <= mmax - nd; nc++)
                                                        kk[nd][nc] = kk[nd - 1][nc + 1] + rcd[d] * kk[nd - 1][nc];
                                        bvec *Eij = E + (i * NJ + j) * NK * NL;
                                        for (int k = 0; k <= lK; k++)
                                                for (int l = 0; l <= ll; l++)
                                                        Eij[k * NL + l] = kbase ? kk[l][k] : kk[k][l];
                                }
                        }
                }

                // g0, g1 = nabla_J g0 (K, I), g2 = nabla_I g0, g3 = nabla_I g1 at the output's powers
                const bvec ai2 = -2. * ai;
                const bvec a2 = kind == HESS_IPVIP1 ? -2. * aj : kind == HESS_IP1IP2 ? -2. * ak : ai2;
                const int s1 = kind == HESS_IPVIP1 ? sJ : kind == HESS_IP1IP2 ? sK : sI;
                for (int r = 0; r < nroots; r++)
                for (int d = 0; d < 3; d++) {
                        const bvec *E = g2d + (r * 3 + d) * nE;
                        bvec *o0 = G0 + (r * 3 + d) * nO;
                        bvec *o1 = G1 + (r * 3 + d) * nO;
                        bvec *o2 = G2 + (r * 3 + d) * nO;
                        bvec *o3 = G3 + (r * 3 + d) * nO;
                        int q = 0;
                        for (int i = 0; i <= li; i++)
                        for (int j = 0; j <= lj; j++)
                        for (int k = 0; k <= lk; k++)
                        for (int l = 0; l <= ll; l++, q++) {
                                const int p = ((i * NJ + j) * NK + k) * NL + l;
                                // the power of the centre of g1's nabla
                                const int n1 = kind == HESS_IPVIP1 ? j : kind == HESS_IP1IP2 ? k : i;
                                o0[q] = E[p];
                                o1[q] = _hess_nabla(E, p, s1, n1, a2);
                                o2[q] = _hess_nabla(E, p, sI, i, ai2);
                                // g1 at i+1 and i-1 (for ipip1 its power moves with i)
                                const bvec up = _hess_nabla(E, p + sI, s1, kind == HESS_IPIP1 ? i + 1 : n1, a2);
                                if (i) {
                                        const bvec dn = _hess_nabla(E, p - sI, s1, kind == HESS_IPIP1 ? i - 1 : n1, a2);
                                        o3[q] = i * dn + ai2 * up;
                                } else {
                                        o3[q] = ai2 * up;
                                }
                        }
                }

                bvec *o = gc;
                for (int lc = 0; lc < ncl; lc++)
                for (int kc = 0; kc < nck; kc++)
                for (int jc = 0; jc < ncj; jc++)
                for (int ic = 0; ic < nci; ic++) {
                        const int x = ox[0][ic] + ox[1][jc] + ox[2][kc] + ox[3][lc];
                        const int y = oy[0][ic] + oy[1][jc] + oy[2][kc] + oy[3][lc] + nO;
                        const int z = oz[0][ic] + oz[1][jc] + oz[2][kc] + oz[3][lc] + 2 * nO;
                        bvec s[9];
                        for (int c = 0; c < 9; c++) s[c] = (bvec){0};
                        for (int r = 0; r < nroots; r++) {
                                const int rb = r * 3 * nO;
                                const bvec *g0 = G0 + rb, *g1 = G1 + rb, *g2 = G2 + rb, *g3 = G3 + rb;
                                s[0] += g3[x] * g0[y] * g0[z];
                                s[1] += g2[x] * g1[y] * g0[z];
                                s[2] += g2[x] * g0[y] * g1[z];
                                s[3] += g1[x] * g2[y] * g0[z];
                                s[4] += g0[x] * g3[y] * g0[z];
                                s[5] += g0[x] * g2[y] * g1[z];
                                s[6] += g1[x] * g0[y] * g2[z];
                                s[7] += g0[x] * g1[y] * g2[z];
                                s[8] += g0[x] * g0[y] * g3[z];
                        }
                        if (kind == HESS_IPIP1) {
                                // the order of autocode/hess.c's int2e_ipip1
                                o[0 * ncart] += s[0];
                                o[1 * ncart] += s[3];
                                o[2 * ncart] += s[6];
                                o[3 * ncart] += s[1];
                                o[4 * ncart] += s[4];
                                o[5 * ncart] += s[7];
                                o[6 * ncart] += s[2];
                                o[7 * ncart] += s[5];
                                o[8 * ncart] += s[8];
                        } else {
                                for (int c = 0; c < 9; c++) o[c * ncart] += s[c];
                        }
                        o++;
                }
        }

        if (dot) {
                // gam of the lanes to cartesian, [ng][ncl][nck][ncj][nci], in the space of the 2D
                // integrals (free now; 2 ncart is at most 0.6 of that space of a class, for
                // ipip1 (ff|ff); t1 and t2 for the passes); lanes past nq take 0. Then the sums
                // with the cartesian integrals.
                bvec *gcart = g2d;
                for (int n = 0; n < ng; n++) {
                        for (int f = 0; f < nf; f++) {
                                bvec v = {0};
                                for (int q = 0; q < nq; q++) v[q] = gam[((size_t)q * ng + n) * nf + f];
                                t2[f] = v;
                        }
                        const bvec *src = t2;
                        bvec *dst;
                        if (li >= 2) {
                                _batch_s2c(t1, src, li, nsl * nsk * nsj, 1);
                                src = t1;
                        }
                        if (lj >= 2) {
                                dst = src == t1 ? t2 : t1;
                                _batch_s2c(dst, src, lj, nsl * nsk, nci);
                                src = dst;
                        }
                        if (lk >= 2) {
                                dst = src == t1 ? t2 : t1;
                                _batch_s2c(dst, src, lk, nsl, ncj * nci);
                                src = dst;
                        }
                        bvec *gn = gcart + n * ncart;
                        if (ll >= 2) {
                                _batch_s2c(gn, src, ll, 1, nck * ncj * nci);
                        } else {
                                for (int f = 0; f < ncart; f++) gn[f] = src[f];
                        }
                }
                // one pass over the cartesian integrals (5.8 MB for (ff|ff) with 8 lanes), the sums in
                // registers
                bvec s0[9], s1[9];
                for (int c = 0; c < 9; c++) {
                        s0[c] = (bvec){0};
                        s1[c] = (bvec){0};
                }
                if (ng == 2) {
                        const bvec *g0 = gcart, *g1 = gcart + ncart;
                        for (int f = 0; f < ncart; f++) {
                                const bvec a = g0[f], a1 = g1[f];
                                for (int c = 0; c < 9; c++) {
                                        const bvec x = gc[c * ncart + f];
                                        s0[c] += x * a;
                                        s1[c] += x * a1;
                                }
                        }
                } else {
                        for (int f = 0; f < ncart; f++) {
                                const bvec a = gcart[f];
                                for (int c = 0; c < 9; c++) s0[c] += gc[c * ncart + f] * a;
                        }
                }
                for (int q = 0; q < nq; q++)
                for (int c = 0; c < 9; c++) {
                        out[(size_t)q * ng * 9 + c] = s0[c][q];
                        if (ng == 2) out[((size_t)q * ng + 1) * 9 + c] = s1[c][q];
                }
                return;
        }

        // cart -> sph over l, k, j, i in turn, one component at a time:
        // [lc][kc][jc][ic] -> [ls][ks][js][is]
        for (int c = 0; c < 9; c++) {
                const bvec *src = gc + c * ncart;
                bvec *dst;
                if (ll >= 2) {
                        _batch_c2s(t1, src, ll, 1, nck * ncj * nci);
                        src = t1;
                }
                if (lk >= 2) {
                        dst = src == t1 ? t2 : t1;
                        _batch_c2s(dst, src, lk, nsl, ncj * nci);
                        src = dst;
                }
                if (lj >= 2) {
                        dst = src == t1 ? t2 : t1;
                        _batch_c2s(dst, src, lj, nsl * nsk, nci);
                        src = dst;
                }
                if (li >= 2) {
                        dst = src == t1 ? t2 : t1;
                        _batch_c2s(dst, src, li, nsl * nsk * nsj, 1);
                        src = dst;
                }
                for (int q = 0; q < nq; q++) {
                        double *oq = out + ((size_t)q * 9 + c) * nf;
                        for (int n = 0; n < nf; n++) oq[n] = src[n][q];
                }
        }
}

#define HESS_CASE(LI, J, K, L, KIND) case (J) << 4 | (K) << 2 | (L): \
        CINTbatch2e_hess_t(out, nq, b, omega, cache, LI, J, K, L, KIND, DOT, gam, ng); break;
#define HESS_L(LI, J, K, KIND) HESS_CASE(LI, J, K, 0, KIND) HESS_CASE(LI, J, K, 1, KIND) \
        HESS_CASE(LI, J, K, 2, KIND) HESS_CASE(LI, J, K, 3, KIND)
#define HESS_K(LI, J, KIND) HESS_L(LI, J, 0, KIND) HESS_L(LI, J, 1, KIND) HESS_L(LI, J, 2, KIND) \
        HESS_L(LI, J, 3, KIND)
// the integrals, and the _dot kernels (DOT, gam and ng are the names HESS_CASE takes)
#define HESS_DEFINE_LI(NAME, LI, KIND) \
BATCH_HIDDEN void CINTbatch2e_##NAME##_l##LI(double *out, int nq, CINTBatch2e *b, double omega, \
                                             bvec *cache, int lj, int lk, int ll) \
{ \
        const int DOT = 0; \
        const double *gam = NULL; \
        const int ng = 0; \
        switch (lj << 4 | lk << 2 | ll) { \
        HESS_K(LI, 0, KIND) HESS_K(LI, 1, KIND) HESS_K(LI, 2, KIND) HESS_K(LI, 3, KIND) \
        } \
} \
BATCH_HIDDEN void CINTbatch2e_##NAME##_dot_l##LI(double *out, int nq, CINTBatch2e *b, double omega, \
                                                 bvec *cache, int lj, int lk, int ll, \
                                                 const double *gam, int ng) \
{ \
        const int DOT = 1; \
        switch (lj << 4 | lk << 2 | ll) { \
        HESS_K(LI, 0, KIND) HESS_K(LI, 1, KIND) HESS_K(LI, 2, KIND) HESS_K(LI, 3, KIND) \
        } \
}
