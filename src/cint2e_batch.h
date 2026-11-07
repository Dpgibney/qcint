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
 * int2e_sph of up to SIMDD shell quartets of one class at once, the quartets in the SIMD lanes.
 * int2e_sph puts the primitive combinations of one quartet in its lanes, so a quartet of
 * single-primitive shells (the d and f shells of def2-TZVP(D)) runs on its scalar path there.
 * Here the class (l of each shell) is a compile-time constant of an always-inlined template,
 * instantiated for every l <= 3 in cint2e_batch_l*.c, and every lane holds its own quartet:
 * its primitive combinations are summed in turn. The recurrences are those of int2e_sph's
 * scalar path (g2e_simd1.c): roots and coefficients, the vertical recurrence on the same base
 * centres (i if li > lj, else j; k if lk > ll, else l), horizontal transfers to the others, the
 * cartesian products, and the cart -> sph transform.
 */
#include <math.h>
#include <string.h>
#include "cint.h"
#include "rys_roots.h"

typedef double bvec __attribute__((vector_size(SIMDD * 8), aligned(SIMDD * 8)));

#define BATCH_MAXL      3
#define BATCH_MAXP      7       // primitives of a shell
#define BATCH_MAXR      7       // Rys roots of (ff|ff), and of its first derivative
// workspace in bvec, for int2e (one component) and int2e_ip1 (three, li+1 on i): 2D integrals
// [roots][3][(li+1+deriv)(lj+1)(lk+1)(ll+1)], cartesian integrals [comp][...], and two arrays of the
// cart -> sph transform
#define BATCH_NI_MAX    (BATCH_MAXR * 3 * 5 * 4 * 4 * 4)
#define BATCH_NC_MAX    (3 * 10000)
#define BATCH_NT1_MAX   (3 * 7000)
#define BATCH_NT2_MAX   (3 * 4900)
#define BATCH_CACHE_BVEC (BATCH_NI_MAX + BATCH_NC_MAX + BATCH_NT1_MAX + BATCH_NT2_MAX)

typedef struct {
        int np[4];
        bvec a[4][BATCH_MAXP];  // exponents, per lane
        bvec c[4][BATCH_MAXP];  // coefficients (contraction and normalization), per lane
        bvec r[4][3];           // centres, per lane
} CINTBatch2e;

#define BATCH_HIDDEN __attribute__((visibility("hidden")))
BATCH_HIDDEN void CINTbatch2e_l0(double *out, int nq, CINTBatch2e *b, double omega, bvec *cache, int lj, int lk, int ll);
BATCH_HIDDEN void CINTbatch2e_l1(double *out, int nq, CINTBatch2e *b, double omega, bvec *cache, int lj, int lk, int ll);
BATCH_HIDDEN void CINTbatch2e_l2(double *out, int nq, CINTBatch2e *b, double omega, bvec *cache, int lj, int lk, int ll);
BATCH_HIDDEN void CINTbatch2e_l3(double *out, int nq, CINTBatch2e *b, double omega, bvec *cache, int lj, int lk, int ll);
// (the int2e_ip1 kernels, CINTbatch2e_ip1_l*, come from cint2e_batch_hess.h)

// libcint's order of the cartesian functions
static const int BATCH_CART[BATCH_MAXL + 1][10][3] = {
        {{0, 0, 0}},
        {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
        {{2, 0, 0}, {1, 1, 0}, {1, 0, 1}, {0, 2, 0}, {0, 1, 1}, {0, 0, 2}},
        {{3, 0, 0}, {2, 1, 0}, {2, 0, 1}, {1, 2, 0}, {1, 1, 1}, {1, 0, 2}, {0, 3, 0}, {0, 2, 1}, {0, 1, 2},
         {0, 0, 3}}};
// libcint's cart -> sph for d and f (s and p are identities; their factors are in the prefactor)
static const double BATCH_C2S[BATCH_MAXL + 1][7][10] = {
        {{1}},
        {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
        {{0.0, 1.0925484305920792, 0.0, 0.0, 0.0, 0.0},
         {0.0, 0.0, 0.0, 0.0, 1.0925484305920792, 0.0},
         {-0.31539156525252, 0.0, 0.0, -0.31539156525252, 0.0, 0.63078313050504},
         {0.0, 0.0, 1.0925484305920792, 0.0, 0.0, 0.0},
         {0.5462742152960396, 0.0, 0.0, -0.5462742152960396, 0.0, 0.0}},
        {{0.0, 1.7701307697799304, 0.0, 0.0, 0.0, 0.0, -0.5900435899266435, 0.0, 0.0, 0.0},
         {0.0, 0.0, 0.0, 0.0, 2.8906114426405543, 0.0, 0.0, 0.0, 0.0, 0.0},
         {0.0, -0.4570457994644657, 0.0, 0.0, 0.0, 0.0, -0.4570457994644657, 0.0, 1.8281831978578629, 0.0},
         {0.0, 0.0, -1.1195289977703462, 0.0, 0.0, 0.0, 0.0, -1.1195289977703462, 0.0, 0.7463526651802308},
         {-0.4570457994644657, 0.0, 0.0, -0.4570457994644657, 0.0, 1.8281831978578629, 0.0, 0.0, 0.0, 0.0},
         {0.0, 0.0, 1.4453057213202771, 0.0, 0.0, 0.0, 0.0, -1.4453057213202771, 0.0, 0.0},
         {0.5900435899266435, 0.0, 0.0, -1.7701307697799304, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}};

static inline double _batch_fac_sp(int l)
{
        return l == 0 ? 0.282094791773878143 : l == 1 ? 0.488602511902919921 : 1.;
}

/*
 * One cart -> sph pass over the index of stride `inner` and extent nc (-> ns), for every combination of
 * `outer` outer and `inner` inner elements: dst[o][s][x] = sum_c C2S[l][s][c] src[o][c][x]
 */
static inline __attribute__((always_inline))
void _batch_c2s(bvec *dst, const bvec *src, const int l, int outer, int inner)
{
        const int nc = (l + 1) * (l + 2) / 2, ns = 2 * l + 1;
        for (int o = 0; o < outer; o++)
        for (int s = 0; s < ns; s++) {
                bvec *d = dst + (o * ns + s) * inner;
                const bvec *x = src + o * nc * inner;
                for (int n = 0; n < inner; n++) d[n] = (bvec){0};
                for (int c = 0; c < nc; c++) {
                        const double f = BATCH_C2S[l][s][c];
                        if (f != 0) {
                                for (int n = 0; n < inner; n++) d[n] += f * x[c * inner + n];
                        }
                }
        }
}

static inline __attribute__((always_inline))
void CINTbatch2e_t(double *out, int nq, CINTBatch2e *b, double omega, bvec *cache,
                   const int li, const int lj, const int lk, const int ll, const int deriv)
{
        // deriv = 1: int2e_ip1, the gradient of the first shell's functions, as libcint's nabla1i,
        // d/dx f_i = i f_{i-1} - 2 a_i f_{i+1}: the 2D integrals of i go one higher. No longer
        // instantiated (int2e_ip1_sph_batch comes from cint2e_batch_hess.h with the Hessians'
        // integrals). The branch stays so that int2e's kernels compile from the same source: without
        // it GCC contracted other multiply-adds into FMAs, and 501 of benzene/def2-SVP's 969 classes
        // changed in their last bits.
        const int lI = li + deriv;
        const int ncomp = deriv ? 3 : 1;
        const int nroots = (lI + lj + lk + ll) / 2 + 1;
        const int nmax = lI + lj, mmax = lk + ll;
        const int Ni = lI + 1, Nj = lj + 1, Nk = lk + 1, Nl = ll + 1;
        const int nI = Ni * Nj * Nk * Nl;
        const int nci = (li + 1) * (li + 2) / 2, ncj = (lj + 1) * (lj + 2) / 2;
        const int nck = (lk + 1) * (lk + 2) / 2, ncl = (ll + 1) * (ll + 2) / 2;
        const int nsi = 2 * li + 1, nsj = 2 * lj + 1, nsk = 2 * lk + 1, nsl = 2 * ll + 1;
        const int ncart = nci * ncj * nck * ncl;
        const int nf = nsi * nsj * nsk * nsl;
        bvec *g2d = cache;                      // [nroots][3][Ni][Nj][Nk][Nl]
        bvec *gc = g2d + nroots * 3 * nI;       // [comp][ncl][nck][ncj][nci]
        bvec *t1 = gc + ncomp * ncart;
        bvec *t2 = t1 + BATCH_NT1_MAX;
        const double common = M_PI * M_PI * M_PI * 2 / sqrt(M_PI)
                * _batch_fac_sp(li) * _batch_fac_sp(lj) * _batch_fac_sp(lk) * _batch_fac_sp(ll);
        // offsets of the powers of each cartesian function in g2d's [Ni][Nj][Nk][Nl]
        int ox[4][10], oy[4][10], oz[4][10];
        const int ls[4] = {li, lj, lk, ll};
        const int stride[4] = {Nj * Nk * Nl, Nk * Nl, Nl, 1};
        for (int t = 0; t < 4; t++) {
                for (int c = 0; c < (ls[t] + 1) * (ls[t] + 2) / 2; c++) {
                        ox[t][c] = BATCH_CART[ls[t]][c][0] * stride[t];
                        oy[t][c] = BATCH_CART[ls[t]][c][1] * stride[t];
                        oz[t][c] = BATCH_CART[ls[t]][c][2] * stride[t];
                }
        }

        // the base centres of the recurrences, as int2e_sph and int2e_ip1: a = i if li(+1) > lj, else j;
        // c = k if lk > ll, else l
        const int ibase = lI > lj, kbase = lk > ll;
        const int lb = ibase ? lj : lI, ld = kbase ? ll : lk;
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
        for (int n = 0; n < ncomp * ncart; n++) gc[n] = (bvec){0};

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
                double u[BATCH_MAXR][SIMDD] __attribute__((aligned(SIMDD * 8)));
                double w[BATCH_MAXR][SIMDD] __attribute__((aligned(SIMDD * 8)));
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
                // the lanes' roots at once where they allow it (all at large x, or 1 or 2 roots), as int2e_sph
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
                                bvec g[2 * BATCH_MAXL + 2][2 * BATCH_MAXL + 1];  // [n][m]: (n 0| m 0)
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
                                bvec h[BATCH_MAXL + 2][2 * BATCH_MAXL + 2][2 * BATCH_MAXL + 1];  // [b][a][m]
                                for (int n = 0; n <= nmax; n++)
                                        for (int m = 0; m <= mmax; m++) h[0][n][m] = g[n][m];
                                for (int nb = 1; nb <= lb; nb++)
                                        for (int na = 0; na <= nmax - nb; na++)
                                                for (int m = 0; m <= mmax; m++)
                                                        h[nb][na][m] = h[nb - 1][na + 1][m] + rab[d] * h[nb - 1][na][m];
                                bvec *Id = g2d + (r * 3 + d) * nI;
                                for (int i = 0; i <= lI; i++)
                                for (int j = 0; j <= lj; j++) {
                                        // |c, d+1) = |c+1, d) + R_cd |c, d), c the base centre of the ket
                                        const bvec *row = ibase ? h[j][i] : h[i][j];
                                        bvec kk[BATCH_MAXL + 1][2 * BATCH_MAXL + 1];  // [d][c]
                                        for (int m = 0; m <= mmax; m++) kk[0][m] = row[m];
                                        for (int nd = 1; nd <= ld; nd++)
                                                for (int nc = 0; nc <= mmax - nd; nc++)
                                                        kk[nd][nc] = kk[nd - 1][nc + 1] + rcd[d] * kk[nd - 1][nc];
                                        bvec *Iij = Id + (i * Nj + j) * Nk * Nl;
                                        for (int k = 0; k <= lk; k++)
                                                for (int l = 0; l <= ll; l++)
                                                        Iij[k * Nl + l] = kbase ? kk[l][k] : kk[k][l];
                                }
                        }
                }
                const bvec ai2 = -2. * ai;
                const int si = stride[0];
                bvec *o = gc;
                for (int lc = 0; lc < ncl; lc++)
                for (int kc = 0; kc < nck; kc++)
                for (int jc = 0; jc < ncj; jc++)
                for (int ic = 0; ic < nci; ic++) {
                        const int x = ox[0][ic] + ox[1][jc] + ox[2][kc] + ox[3][lc];
                        const int y = oy[0][ic] + oy[1][jc] + oy[2][kc] + oy[3][lc] + nI;
                        const int z = oz[0][ic] + oz[1][jc] + oz[2][kc] + oz[3][lc] + 2 * nI;
                        if (deriv == 0) {
                                bvec s = *o;
                                for (int r = 0; r < nroots; r++) {
                                        const bvec *Ir = g2d + r * 3 * nI;
                                        s += Ir[x] * Ir[y] * Ir[z];
                                }
                                *o = s;
                        } else {
                                const int ix = BATCH_CART[li][ic][0], iy = BATCH_CART[li][ic][1];
                                const int iz = BATCH_CART[li][ic][2];
                                bvec sx = o[0], sy = o[ncart], sz = o[2 * ncart];
                                for (int r = 0; r < nroots; r++) {
                                        const bvec *Ir = g2d + r * 3 * nI;
                                        const bvec gx = Ir[x], gy = Ir[y], gz = Ir[z];
                                        bvec fx = ai2 * Ir[x + si];
                                        bvec fy = ai2 * Ir[y + si];
                                        bvec fz = ai2 * Ir[z + si];
                                        if (ix) fx = ix * Ir[x - si] + fx;
                                        if (iy) fy = iy * Ir[y - si] + fy;
                                        if (iz) fz = iz * Ir[z - si] + fz;
                                        sx += fx * gy * gz;
                                        sy += gx * fy * gz;
                                        sz += gx * gy * fz;
                                }
                                o[0] = sx;
                                o[ncart] = sy;
                                o[2 * ncart] = sz;
                        }
                        o++;
                }
        }

        // cart -> sph over l, k, j, i in turn: [comp][lc][kc][jc][ic] -> [comp][ls][ks][js][is]
        const bvec *src = gc;
        bvec *dst;
        if (ll >= 2) {
                _batch_c2s(t1, src, ll, ncomp, nck * ncj * nci);
                src = t1;
        }
        if (lk >= 2) {
                dst = src == t1 ? t2 : t1;
                _batch_c2s(dst, src, lk, ncomp * nsl, ncj * nci);
                src = dst;
        }
        if (lj >= 2) {
                dst = src == t1 ? t2 : t1;
                _batch_c2s(dst, src, lj, ncomp * nsl * nsk, nci);
                src = dst;
        }
        if (li >= 2) {
                dst = src == t1 ? t2 : t1;
                _batch_c2s(dst, src, li, ncomp * nsl * nsk * nsj, 1);
                src = dst;
        }
        for (int q = 0; q < nq; q++) {
                double *oq = out + (size_t)q * ncomp * nf;
                for (int n = 0; n < ncomp * nf; n++) oq[n] = src[n][q];
        }
}

#define BATCH_CASE(LI, J, K, L, D) case (J) << 4 | (K) << 2 | (L): \
        CINTbatch2e_t(out, nq, b, omega, cache, LI, J, K, L, D); break;
#define BATCH_L(LI, J, K, D) BATCH_CASE(LI, J, K, 0, D) BATCH_CASE(LI, J, K, 1, D) BATCH_CASE(LI, J, K, 2, D) \
        BATCH_CASE(LI, J, K, 3, D)
#define BATCH_K(LI, J, D) BATCH_L(LI, J, 0, D) BATCH_L(LI, J, 1, D) BATCH_L(LI, J, 2, D) BATCH_L(LI, J, 3, D)
#define BATCH_DEFINE_LI(LI) \
BATCH_HIDDEN void CINTbatch2e_l##LI(double *out, int nq, CINTBatch2e *b, double omega, bvec *cache, \
                                    int lj, int lk, int ll) \
{ \
        switch (lj << 4 | lk << 2 | ll) { \
        BATCH_K(LI, 0, 0) BATCH_K(LI, 1, 0) BATCH_K(LI, 2, 0) BATCH_K(LI, 3, 0) \
        } \
}
