/*
 * Copyright (C) 2021  Qiming Sun <osirpt.sun@gmail.com>
 *
 */

#include <string.h>
#include <math.h>
#include <assert.h>
#include "cint_bas.h"
#include "misc.h"
#include "g1e.h"
#include "g1e_grids.h"
#include "rys_roots.h"
#include "rys_roots_chunk.h"
#include "simd.h"


void CINTinit_int1e_grids_EnvVars(CINTEnvVars *envs, int *ng, int *shls,
                                  int *atm, int natm,
                                  int *bas, int nbas, double *env)
{
        CINTinit_int1e_EnvVars(envs, ng, shls, atm, natm, bas, nbas, env);
        envs->opt = NULL;
        int ngrids = shls[3] - shls[2];
        double *grids = env + (size_t)env[PTR_GRIDS] + shls[2] * 3;

        envs->ngrids = ngrids;
        envs->grids = grids;
        envs->common_factor = 2 * M_PI
                * CINTcommon_fac_sp(envs->i_l) * CINTcommon_fac_sp(envs->j_l);

        int rys_order = envs->nrys_roots;
        int nroots = rys_order;
        double omega = env[PTR_RANGE_OMEGA];
        if (omega < 0 && rys_order <= 3) {
                nroots *= 2;
        } else if (CINTgrids_mixed_range(env)) {
                // the roots of the full-range and of the long-range operator
                nroots *= 2;
        }
        envs->rys_order = rys_order;
        envs->nrys_roots = nroots;

        int dli, dlj;
        int ibase = envs->li_ceil > envs->lj_ceil;
        if (ibase) {
                dli = envs->li_ceil + envs->lj_ceil + 1;
                dlj = envs->lj_ceil + 1;
                envs->rirj[0] = envs->ri[0] - envs->rj[0];
                envs->rirj[1] = envs->ri[1] - envs->rj[1];
                envs->rirj[2] = envs->ri[2] - envs->rj[2];
        } else {
                dli = envs->li_ceil + 1;
                dlj = envs->li_ceil + envs->lj_ceil + 1;
                envs->rirj[0] = envs->rj[0] - envs->ri[0];
                envs->rirj[1] = envs->rj[1] - envs->ri[1];
                envs->rirj[2] = envs->rj[2] - envs->ri[2];
        }
        envs->g_stride_i = GRID_BLKSIZE * nroots;
        envs->g_stride_j = GRID_BLKSIZE * nroots * dli;
        envs->g_size     = GRID_BLKSIZE * nroots * dli * dlj;
        envs->g_stride_k = envs->g_size;
        envs->g_stride_l = envs->g_size;
}

#define RGSQUARE(r, ig)     (r[ig+GRID_BLKSIZE*0]*r[ig+GRID_BLKSIZE*0] + \
                             r[ig+GRID_BLKSIZE*1]*r[ig+GRID_BLKSIZE*1] + \
                             r[ig+GRID_BLKSIZE*2]*r[ig+GRID_BLKSIZE*2])

/*
 * Whether env selects the mixed-range operator c_full/r12 + c_lr erf(omega r12)/r12 of the int1e_grids
 * integrals (PTR_RANGE_COEFF_FULL, PTR_RANGE_COEFF_LR)
 */
int CINTgrids_mixed_range(double *env)
{
        return (env[PTR_RANGE_OMEGA] > 0 && env[PTR_RINV_ZETA] == 0 &&
                (env[PTR_RANGE_COEFF_FULL] != 0 || env[PTR_RANGE_COEFF_LR] != 0));
}

/*
 * The Rys roots and weights of one Coulomb-type operator on the points of a block, for the mixed-range
 * operator: u[ig+GRID_BLKSIZE*i] = t^2 * theta and w[ig+GRID_BLKSIZE*i] = weight * fac1 of the nroots roots at
 * x = a0 |r_ij - r_g|^2. The bare Coulomb operator is a0 = aij, theta = 1; the long-range one a0 = aij*theta
 * with sqrt(theta) in fac1. The chunks of SIMDD points are those of CINTg0_1e_grids.
 */
static void __attribute__((noinline))
_grids_roots_set(double *RESTRICT u, double *RESTRICT w, double *RESTRICT rijrg, int bgrids,
                 int nroots, double a0, double fac1, double theta)
{
        double ubuf[MXRYSROOTS];
        double wbuf[MXRYSROOTS];
        double x;
        int i, ig;
#ifdef HAVE_RYS_ROOTS_CHUNK
        const double *rt_large, *ww_large;
        CINTrys_largex_tables(nroots, &rt_large, &ww_large);
        const __MD rlarge = MM_SET1(35. + nroots * 5);
        const __MD rpie4 = MM_SET1(CHUNK_PIE4);
        const __MD ra0 = MM_SET1(a0);
        const __MD rfac1 = MM_SET1(fac1);
        const __MD rtheta = MM_SET1(theta);
#endif
        for (ig = 0; ig < bgrids; ig += SIMDD) {
#ifdef HAVE_RYS_ROOTS_CHUNK
                if (ig + SIMDD <= bgrids) {
                        __MD dx = MM_LOAD(rijrg+ig+GRID_BLKSIZE*0);
                        __MD dy = MM_LOAD(rijrg+ig+GRID_BLKSIZE*1);
                        __MD dz = MM_LOAD(rijrg+ig+GRID_BLKSIZE*2);
                        __MD rx = ra0 * (dx * dx + dy * dy + dz * dz);
                        __MD rt2_1, rw_1;
                        if (nroots == 1 && CINTrys_root1_chunk(rx, &rt2_1, &rw_1, 1)) {
                                MM_STORE(u+ig, rt2_1 * rtheta);
                                MM_STORE(w+ig, rw_1 * rfac1);
                                continue;
                        }
                        if (MM_CMP(rx, rlarge, _CMP_GE_OQ) == CHUNK_LANES) {
                                __MD rt = MM_SQRT(rpie4 / rx);
                                __MD rt2 = rt * rt * rtheta;
                                for (i = 0; i < nroots; i++) {
                                        MM_STORE(u+ig+GRID_BLKSIZE*i, MM_SET1(rt_large[i] / CHUNK_PIE4) * rt2);
                                        MM_STORE(w+ig+GRID_BLKSIZE*i, MM_SET1(ww_large[i]) * rt * rfac1);
                                }
                                continue;
                        }
                        if (nroots == 2) {
                                ALIGNMM double xs[SIMDD];
                                ALIGNMM double us[SIMDD*2];
                                ALIGNMM double ws[SIMDD*2];
                                MM_STORE(xs, rx);
                                if (CINTrys_roots_chunk(2, xs, us, ws)) {
                                        for (i = 0; i < 2; i++) {
                                                __MD ru = MM_LOAD(us+SIMDD*i);
                                                MM_STORE(u+ig+GRID_BLKSIZE*i, ru / (ru + MM_SET1(1.)) * rtheta);
                                                MM_STORE(w+ig+GRID_BLKSIZE*i, MM_LOAD(ws+SIMDD*i) * rfac1);
                                        }
                                        continue;
                                }
                        }
                }
#endif
                int ig1 = MIN(ig + SIMDD, bgrids);
                for (int jg = ig; jg < ig1; jg++) {
                        x = a0 * RGSQUARE(rijrg, jg);
                        CINTrys_roots(nroots, x, ubuf, wbuf);
                        for (i = 0; i < nroots; i++) {
                                u[jg+GRID_BLKSIZE*i] = ubuf[i] / (ubuf[i] + 1) * theta;
                                w[jg+GRID_BLKSIZE*i] = wbuf[i] * fac1;
                        }
                }
        }
}

int CINTg0_1e_grids(double *RESTRICT g, double cutoff, CINTEnvVars *envs,
                    double *cache, double *RESTRICT gridsT)
{
        int ngrids = envs->ngrids;
        int bgrids = MIN(ngrids - envs->grids_offset, GRID_BLKSIZE);
        int nroots = envs->nrys_roots;
        double *RESTRICT gx = g;
        double *RESTRICT gy = g + envs->g_size;
        double *RESTRICT gz = g + envs->g_size * 2;
        double *RESTRICT w = gz;
        double *rij = envs->rij;
        double ubuf[MXRYSROOTS];
        double wbuf[MXRYSROOTS];
        double *RESTRICT u;
        MALLOC_ALIGNED_DOUBLE_INSTACK(u, GRID_BLKSIZE*nroots);
        double *RESTRICT rijrg;
        MALLOC_ALIGNED_DOUBLE_INSTACK(rijrg, GRID_BLKSIZE*3);
        double aij = envs->ai[0] + envs->aj[0];
        int n, i, j, ig;
        double x, fac1;

        __MD r0, r1, r2;
        r1 = MM_SET1(1.);
        for (i = 0; i < nroots; i++) {
                for (ig = 0; ig < bgrids; ig += SIMDD) {
                        MM_STORE(gx+ig+GRID_BLKSIZE*i, r1);
                        MM_STORE(gy+ig+GRID_BLKSIZE*i, r1);
                }
        }

        r0 = MM_SET1(rij[0]);
        r1 = MM_SET1(rij[1]);
        r2 = MM_SET1(rij[2]);
        for (ig = 0; ig < bgrids; ig += SIMDD) {
                MM_STORE(rijrg+ig+GRID_BLKSIZE*0, MM_LOAD(gridsT+ig+GRID_BLKSIZE*0) - r0);
                MM_STORE(rijrg+ig+GRID_BLKSIZE*1, MM_LOAD(gridsT+ig+GRID_BLKSIZE*1) - r1);
                MM_STORE(rijrg+ig+GRID_BLKSIZE*2, MM_LOAD(gridsT+ig+GRID_BLKSIZE*2) - r2);
        }

        double omega = envs->env[PTR_RANGE_OMEGA];
        double zeta = envs->env[PTR_RINV_ZETA];
        double omega2, theta, sqrt_theta, a0, tau2;

        assert(zeta >= 0);
        if (omega == 0. && zeta == 0.) {
                fac1 = envs->fac[0] / aij;
#ifdef HAVE_RYS_ROOTS_CHUNK
                // Most points are far from the pair (x >= 35+5*nroots), where CINTrys_roots returns
                // u = rt/(x-rt) and w = ww*sqrt(PIE4/x). A chunk of SIMDD points that are all there is done
                // here at once, any chunk of one root by CINTrys_root1_chunk (inlined, t^2 formed directly)
                // and of two by CINTrys_roots_chunk; the others point by point.
                const double *rt_large, *ww_large;
                CINTrys_largex_tables(nroots, &rt_large, &ww_large);
                const __MD rlarge = MM_SET1(35. + nroots * 5);
                const __MD rpie4 = MM_SET1(CHUNK_PIE4);
                const __MD raij = MM_SET1(aij);
                const __MD rfac1 = MM_SET1(fac1);
#endif
                for (ig = 0; ig < bgrids; ig += SIMDD) {
#ifdef HAVE_RYS_ROOTS_CHUNK
                        if (ig + SIMDD <= bgrids) {
                                __MD dx = MM_LOAD(rijrg+ig+GRID_BLKSIZE*0);
                                __MD dy = MM_LOAD(rijrg+ig+GRID_BLKSIZE*1);
                                __MD dz = MM_LOAD(rijrg+ig+GRID_BLKSIZE*2);
                                __MD rx = raij * (dx * dx + dy * dy + dz * dz);
                                __MD rt2_1, rw_1;
                                if (nroots == 1 && CINTrys_root1_chunk(rx, &rt2_1, &rw_1, 1)) {
                                        MM_STORE(u+ig, rt2_1);
                                        MM_STORE(w+ig, rw_1 * rfac1);
                                        continue;
                                }
                                if (MM_CMP(rx, rlarge, _CMP_GE_OQ) == CHUNK_LANES) {
                                        __MD rt = MM_SQRT(rpie4 / rx);
                                        __MD rt2 = rt * rt;
                                        for (i = 0; i < nroots; i++) {
                                                // t^2 = u/(u+1) of u = rt_large/(x-rt_large) is rt_large/x
                                                MM_STORE(u+ig+GRID_BLKSIZE*i, MM_SET1(rt_large[i] / CHUNK_PIE4) * rt2);
                                                MM_STORE(w+ig+GRID_BLKSIZE*i, MM_SET1(ww_large[i]) * rt * rfac1);
                                        }
                                        continue;
                                }
                                if (nroots == 2) {
                                        ALIGNMM double xs[SIMDD];
                                        ALIGNMM double us[SIMDD*2];
                                        ALIGNMM double ws[SIMDD*2];
                                        MM_STORE(xs, rx);
                                        if (CINTrys_roots_chunk(2, xs, us, ws)) {
                                                for (i = 0; i < 2; i++) {
                                                        __MD ru = MM_LOAD(us+SIMDD*i);
                                                        // transform to t^2
                                                        MM_STORE(u+ig+GRID_BLKSIZE*i, ru / (ru + MM_SET1(1.)));
                                                        MM_STORE(w+ig+GRID_BLKSIZE*i, MM_LOAD(ws+SIMDD*i) * rfac1);
                                                }
                                                continue;
                                        }
                                }
                        }
#endif
                        int ig1 = MIN(ig + SIMDD, bgrids);
                        for (int jg = ig; jg < ig1; jg++) {
                                x = aij * RGSQUARE(rijrg, jg);
                                CINTrys_roots(nroots, x, ubuf, wbuf);
                                for (i = 0; i < nroots; i++) {
                                        // transform to t^2
                                        u[jg+GRID_BLKSIZE*i] = ubuf[i] / (ubuf[i] + 1);
                                        w[jg+GRID_BLKSIZE*i] = wbuf[i] * fac1;
                                }
                        }
                }
        } else if (omega < 0.) { // short-range part of range-separated Coulomb
                a0 = aij;
                fac1 = envs->fac[0] / aij;
                if (zeta == 0.) {
                        tau2 = 1.;
                        omega2 = omega * omega;
                        theta = omega2 / (omega2 + aij);
                } else { // zeta > 0.
                        tau2 = zeta / (zeta + aij);
                        a0 *= tau2;
                        fac1 *= sqrt(tau2);
                        omega2 = omega * omega;
                        theta = omega2 / (omega2 + a0);
                }
                sqrt_theta = sqrt(theta);
                // very small erfc() leads to ~0 weights. They can cause
                // numerical issue in sr_rys_roots. Use this cutoff as a
                // temporary solution to avoid numerical issues
                double temp_cutoff = MIN(cutoff, EXPCUTOFF_SR);
                int rorder = envs->rys_order;
                double tau_theta, fac_theta;
                for (ig = 0; ig < bgrids; ig++) {
                        x = a0 * RGSQUARE(rijrg, ig);
                        if (theta * x > temp_cutoff) {
                                // very small erfc() leads to ~0 weights
                                for (i = 0; i < nroots; i++) {
                                        u[ig+GRID_BLKSIZE*i] = 0;
                                        w[ig+GRID_BLKSIZE*i] = 0;
                                }
                        } else if (rorder == nroots) {
                                CINTsr_rys_roots(nroots, x, sqrt_theta, ubuf, wbuf);
                                for (i = 0; i < nroots; i++) {
                                        u[ig+GRID_BLKSIZE*i] = ubuf[i] / (ubuf[i] + 1) * tau2;
                                        w[ig+GRID_BLKSIZE*i] = wbuf[i] * fac1;
                                }
                        } else {
                                tau_theta = tau2 * theta;
                                fac_theta = fac1 * -sqrt_theta;
                                CINTrys_roots(rorder, x, ubuf, wbuf);
                                CINTrys_roots(rorder, theta*x, ubuf+rorder, wbuf+rorder);
                                for (i = 0; i < rorder; i++) {
                                        u[ig+GRID_BLKSIZE*i] = ubuf[i] / (ubuf[i] + 1) * tau2;
                                        w[ig+GRID_BLKSIZE*i] = wbuf[i] * fac1;
                                        u[ig+GRID_BLKSIZE*(i+rorder)] = ubuf[i+rorder] / (ubuf[i+rorder] + 1) * tau_theta;
                                        w[ig+GRID_BLKSIZE*(i+rorder)] = wbuf[i+rorder] * fac_theta;
                                }
                        }
                }
        } else if (CINTgrids_mixed_range(envs->env)) {
                // c_full/r12 + c_lr erf(omega r12)/r12: the roots of the bare Coulomb operator with the
                // weights times c_full, then those of the long-range one with the weights times c_lr
                int rorder = envs->rys_order;
                fac1 = envs->fac[0] / aij;
                omega2 = omega * omega;
                theta = omega2 / (omega2 + aij);
                _grids_roots_set(u, w, rijrg, bgrids, rorder, aij,
                                 fac1 * envs->env[PTR_RANGE_COEFF_FULL], 1.);
                _grids_roots_set(u + GRID_BLKSIZE*rorder, w + GRID_BLKSIZE*rorder, rijrg, bgrids, rorder,
                                 aij * theta, fac1 * sqrt(theta) * envs->env[PTR_RANGE_COEFF_LR], theta);
        } else {
                // * long-range part of range-separated Coulomb
                // * or Gaussian charge model, with rho(r) = Norm exp(-zeta*r^2)
                a0 = aij;
                fac1 = envs->fac[0] / aij;
                if (zeta == 0.) { // omega > 0.
                        omega2 = omega * omega;
                        theta = omega2 / (omega2 + aij);
                        a0 *= theta;
                        fac1 *= sqrt(theta);
                } else if (omega == 0.) { // zeta > 0.
                        theta = zeta / (zeta + aij);
                        a0 *= theta;
                        fac1 *= sqrt(theta);
                } else { // omega > 0. && zeta > 0.
                        omega2 = omega * omega;
                        theta = omega2*zeta / (omega2*zeta + (zeta+omega2)*aij);
                        a0 *= theta;
                        fac1 *= sqrt(theta);
                }
#ifdef HAVE_RYS_ROOTS_CHUNK
                // The chunks of the bare Coulomb branch above, with x = a0*|r_ij-r_g|^2 and t^2 times theta
                const double *rt_large, *ww_large;
                CINTrys_largex_tables(nroots, &rt_large, &ww_large);
                const __MD rlarge = MM_SET1(35. + nroots * 5);
                const __MD rpie4 = MM_SET1(CHUNK_PIE4);
                const __MD ra0 = MM_SET1(a0);
                const __MD rfac1 = MM_SET1(fac1);
                const __MD rtheta = MM_SET1(theta);
#endif
                for (ig = 0; ig < bgrids; ig += SIMDD) {
#ifdef HAVE_RYS_ROOTS_CHUNK
                        if (ig + SIMDD <= bgrids) {
                                __MD dx = MM_LOAD(rijrg+ig+GRID_BLKSIZE*0);
                                __MD dy = MM_LOAD(rijrg+ig+GRID_BLKSIZE*1);
                                __MD dz = MM_LOAD(rijrg+ig+GRID_BLKSIZE*2);
                                __MD rx = ra0 * (dx * dx + dy * dy + dz * dz);
                                __MD rt2_1, rw_1;
                                if (nroots == 1 && CINTrys_root1_chunk(rx, &rt2_1, &rw_1, 1)) {
                                        MM_STORE(u+ig, rt2_1 * rtheta);
                                        MM_STORE(w+ig, rw_1 * rfac1);
                                        continue;
                                }
                                if (MM_CMP(rx, rlarge, _CMP_GE_OQ) == CHUNK_LANES) {
                                        __MD rt = MM_SQRT(rpie4 / rx);
                                        __MD rt2 = rt * rt * rtheta;
                                        for (i = 0; i < nroots; i++) {
                                                MM_STORE(u+ig+GRID_BLKSIZE*i, MM_SET1(rt_large[i] / CHUNK_PIE4) * rt2);
                                                MM_STORE(w+ig+GRID_BLKSIZE*i, MM_SET1(ww_large[i]) * rt * rfac1);
                                        }
                                        continue;
                                }
                                if (nroots == 2) {
                                        ALIGNMM double xs[SIMDD];
                                        ALIGNMM double us[SIMDD*2];
                                        ALIGNMM double ws[SIMDD*2];
                                        MM_STORE(xs, rx);
                                        if (CINTrys_roots_chunk(2, xs, us, ws)) {
                                                for (i = 0; i < 2; i++) {
                                                        __MD ru = MM_LOAD(us+SIMDD*i);
                                                        MM_STORE(u+ig+GRID_BLKSIZE*i, ru / (ru + MM_SET1(1.)) * rtheta);
                                                        MM_STORE(w+ig+GRID_BLKSIZE*i, MM_LOAD(ws+SIMDD*i) * rfac1);
                                                }
                                                continue;
                                        }
                                }
                        }
#endif
                        int ig1 = MIN(ig + SIMDD, bgrids);
                        for (int jg = ig; jg < ig1; jg++) {
                                x = a0 * RGSQUARE(rijrg, jg);
                                CINTrys_roots(nroots, x, ubuf, wbuf);
                                for (i = 0; i < nroots; i++) {
                                        // u stores t^2 = tau^2 * theta
                                        u[jg+GRID_BLKSIZE*i] = ubuf[i] / (ubuf[i] + 1) * theta;
                                        w[jg+GRID_BLKSIZE*i] = wbuf[i] * fac1;
                                }
                        }
                }
        }
        int nmax = envs->li_ceil + envs->lj_ceil;
        if (nmax == 0) {
                return 1;
        }

        double *RESTRICT p0x, *RESTRICT p0y, *RESTRICT p0z;
        double *RESTRICT p1x, *RESTRICT p1y, *RESTRICT p1z;
        double *RESTRICT p2x, *RESTRICT p2y, *RESTRICT p2z;
        int lj, di, dj;
        double *rx;
        if (envs->li_ceil > envs->lj_ceil) {
                // li = envs->li_ceil;
                lj = envs->lj_ceil;
                di = envs->g_stride_i;
                dj = envs->g_stride_j;
                rx = envs->ri;
        } else {
                // li = envs->lj_ceil;
                lj = envs->li_ceil;
                di = envs->g_stride_j;
                dj = envs->g_stride_i;
                rx = envs->rj;
        }
        __MD rijrx = MM_SET1(rij[0] - rx[0]);
        __MD rijry = MM_SET1(rij[1] - rx[1]);
        __MD rijrz = MM_SET1(rij[2] - rx[2]);
        __MD aij2 = MM_SET1(0.5 / aij);
        __MD ru, rt;
        __MD r0x, r0y, r0z;
        __MD r1x, r1y, r1z;
        __MD r2x, r2y, r2z;

        for (n = 0; n < nroots; n++) {
                p0x = gx + GRID_BLKSIZE*n;
                p0y = gy + GRID_BLKSIZE*n;
                p0z = gz + GRID_BLKSIZE*n;
                p1x = p0x + di;
                p1y = p0y + di;
                p1z = p0z + di;
                p2x = p0x - di;
                p2y = p0y - di;
                p2z = p0z - di;
                for (ig = 0; ig < bgrids; ig += SIMDD) {
                        ru = MM_LOAD(u+ig+GRID_BLKSIZE*n);
                        rt = aij2 - aij2 * ru;
                        r0 = rijrx + ru * MM_LOAD(rijrg+ig+GRID_BLKSIZE*0);
                        r1 = rijry + ru * MM_LOAD(rijrg+ig+GRID_BLKSIZE*1);
                        r2 = rijrz + ru * MM_LOAD(rijrg+ig+GRID_BLKSIZE*2);

                        r2x = MM_LOAD(p0x+ig);
                        r2y = MM_LOAD(p0y+ig);
                        r2z = MM_LOAD(p0z+ig);
                        r0x = r0 * r2x;
                        r0y = r1 * r2y;
                        r0z = r2 * r2z;
                        MM_STORE(p1x+ig, r0x);
                        MM_STORE(p1y+ig, r0y);
                        MM_STORE(p1z+ig, r0z);
                        for (i = 1; i < nmax; i++) {
                                //p1x[ig+i*di] = i * rt * p2x[ig+i*di] + r0 * p0x[ig+i*di];
                                //p1y[ig+i*di] = i * rt * p2y[ig+i*di] + r1 * p0y[ig+i*di];
                                //p1z[ig+i*di] = i * rt * p2z[ig+i*di] + r2 * p0z[ig+i*di];
                                r1x = MM_SET1(i) * rt * r2x + r0 * r0x;
                                r1y = MM_SET1(i) * rt * r2y + r1 * r0y;
                                r1z = MM_SET1(i) * rt * r2z + r2 * r0z;
                                MM_STORE(p1x+ig+i*di, r1x);
                                MM_STORE(p1y+ig+i*di, r1y);
                                MM_STORE(p1z+ig+i*di, r1z);
                                r2x = r0x;
                                r2y = r0y;
                                r2z = r0z;
                                r0x = r1x;
                                r0y = r1y;
                                r0z = r1z;
                        }
                }
        }

        __MD rirjx = MM_SET1(envs->rirj[0]);
        __MD rirjy = MM_SET1(envs->rirj[1]);
        __MD rirjz = MM_SET1(envs->rirj[2]);
        for (j = 1; j <= lj; j++) {
        for (n = 0; n < nroots; n++) {
                p0x = gx + j * dj + GRID_BLKSIZE*n;
                p0y = gy + j * dj + GRID_BLKSIZE*n;
                p0z = gz + j * dj + GRID_BLKSIZE*n;
                p1x = p0x - dj;
                p1y = p0y - dj;
                p1z = p0z - dj;
                p2x = p1x + di;
                p2y = p1y + di;
                p2z = p1z + di;

                for (ig = 0; ig < bgrids; ig += SIMDD) {
                        r1x = MM_LOAD(p1x+ig);
                        r1y = MM_LOAD(p1y+ig);
                        r1z = MM_LOAD(p1z+ig);
                        for (i = 0; i <= nmax - j; i++) {
                                //p0x[ig+i*di] = p2x[ig+i*di] + rirjx * p1x[ig+i*di];
                                //p0y[ig+i*di] = p2y[ig+i*di] + rirjy * p1y[ig+i*di];
                                //p0z[ig+i*di] = p2z[ig+i*di] + rirjz * p1z[ig+i*di];
                                r2x = MM_LOAD(p2x+ig+i*di);
                                r2y = MM_LOAD(p2y+ig+i*di);
                                r2z = MM_LOAD(p2z+ig+i*di);
                                MM_STORE(p0x+ig+i*di, r2x + rirjx * r1x);
                                MM_STORE(p0y+ig+i*di, r2y + rirjy * r1y);
                                MM_STORE(p0z+ig+i*di, r2z + rirjz * r1z);
                                r1x = r2x;
                                r1y = r2y;
                                r1z = r2z;
                        }
                }
        } }
        return 1;
}

void CINTgout1e_grids(double *gout, double *g, int *idx,
                      CINTEnvVars *envs, int gout_empty)
{
        int ngrids = envs->ngrids;
        int bgrids = MIN(ngrids - envs->grids_offset, GRID_BLKSIZE);
        bgrids = ALIGN_UP(bgrids, SIMDD);
        int nroots = envs->nrys_roots;
        int nf = envs->nf;
        int i, n, ig;
        double *gx, *gy, *gz;
        __MD s;

        if (gout_empty) {
                switch(nroots) {
                case 1:
                        for (n = 0; n < nf; n++) {
                                gx = g + idx[n*3+0];
                                gy = g + idx[n*3+1];
                                gz = g + idx[n*3+2];
                                for (ig = 0; ig < bgrids; ig += SIMDD) {
                                        s = MM_LOAD(gx+ig) * MM_LOAD(gy+ig) * MM_LOAD(gz+ig);
                                        MM_STORE(gout+ig+bgrids*n, s);
                                }
                        }
                        return;
                case 2:
                        for (n = 0; n < nf; n++) {
                                gx = g + idx[n*3+0];
                                gy = g + idx[n*3+1];
                                gz = g + idx[n*3+2];
                                for (ig = 0; ig < bgrids; ig += SIMDD) {
                                        s  = MM_LOAD(gx+ig             ) * MM_LOAD(gy+ig             ) * MM_LOAD(gz+ig             );
                                        s += MM_LOAD(gx+ig+GRID_BLKSIZE) * MM_LOAD(gy+ig+GRID_BLKSIZE) * MM_LOAD(gz+ig+GRID_BLKSIZE);
                                        MM_STORE(gout+ig+bgrids*n, s);
                                }
                        }
                        break;
                case 3:
                        for (n = 0; n < nf; n++) {
                                gx = g + idx[n*3+0];
                                gy = g + idx[n*3+1];
                                gz = g + idx[n*3+2];
                                for (ig = 0; ig < bgrids; ig += SIMDD) {
                                        s  = MM_LOAD(gx+ig               ) * MM_LOAD(gy+ig               ) * MM_LOAD(gz+ig               );
                                        s += MM_LOAD(gx+ig+GRID_BLKSIZE  ) * MM_LOAD(gy+ig+GRID_BLKSIZE  ) * MM_LOAD(gz+ig+GRID_BLKSIZE  );
                                        s += MM_LOAD(gx+ig+GRID_BLKSIZE*2) * MM_LOAD(gy+ig+GRID_BLKSIZE*2) * MM_LOAD(gz+ig+GRID_BLKSIZE*2);
                                        MM_STORE(gout+ig+bgrids*n, s);
                                }
                        }
                        break;
                case 4:
                        for (n = 0; n < nf; n++) {
                                gx = g + idx[n*3+0];
                                gy = g + idx[n*3+1];
                                gz = g + idx[n*3+2];
                                for (ig = 0; ig < bgrids; ig += SIMDD) {
                                        s  = MM_LOAD(gx+ig               ) * MM_LOAD(gy+ig               ) * MM_LOAD(gz+ig               );
                                        s += MM_LOAD(gx+ig+GRID_BLKSIZE  ) * MM_LOAD(gy+ig+GRID_BLKSIZE  ) * MM_LOAD(gz+ig+GRID_BLKSIZE  );
                                        s += MM_LOAD(gx+ig+GRID_BLKSIZE*2) * MM_LOAD(gy+ig+GRID_BLKSIZE*2) * MM_LOAD(gz+ig+GRID_BLKSIZE*2);
                                        s += MM_LOAD(gx+ig+GRID_BLKSIZE*3) * MM_LOAD(gy+ig+GRID_BLKSIZE*3) * MM_LOAD(gz+ig+GRID_BLKSIZE*3);
                                        MM_STORE(gout+ig+bgrids*n, s);
                                }
                        }
                        break;
                default:
                        for (n = 0; n < nf; n++) {
                                gx = g + idx[n*3+0];
                                gy = g + idx[n*3+1];
                                gz = g + idx[n*3+2];
                                for (ig = 0; ig < bgrids; ig += SIMDD) {
                                        s = MM_SET1(0);
                                        for (i = 0; i < nroots; i++) {
                                                s += MM_LOAD(gx+ig+GRID_BLKSIZE*i) * MM_LOAD(gy+ig+GRID_BLKSIZE*i) * MM_LOAD(gz+ig+GRID_BLKSIZE*i);
                                        }
                                        MM_STORE(gout+ig+bgrids*n, s);
                                }
                        }
                }
        } else {
                switch(nroots) {
                case 1:
                        for (n = 0; n < nf; n++) {
                                gx = g + idx[n*3+0];
                                gy = g + idx[n*3+1];
                                gz = g + idx[n*3+2];
                                for (ig = 0; ig < bgrids; ig += SIMDD) {
                                        s = MM_LOAD(gout+ig+bgrids*n);
                                        s += MM_LOAD(gx+ig) * MM_LOAD(gy+ig) * MM_LOAD(gz+ig);
                                        MM_STORE(gout+ig+bgrids*n, s);
                                }
                        }
                        return;
                case 2:
                        for (n = 0; n < nf; n++) {
                                gx = g + idx[n*3+0];
                                gy = g + idx[n*3+1];
                                gz = g + idx[n*3+2];
                                for (ig = 0; ig < bgrids; ig += SIMDD) {
                                        s = MM_LOAD(gout+ig+bgrids*n);
                                        s += MM_LOAD(gx+ig             ) * MM_LOAD(gy+ig             ) * MM_LOAD(gz+ig             );
                                        s += MM_LOAD(gx+ig+GRID_BLKSIZE) * MM_LOAD(gy+ig+GRID_BLKSIZE) * MM_LOAD(gz+ig+GRID_BLKSIZE);
                                        MM_STORE(gout+ig+bgrids*n, s);
                                }
                        }
                        break;
                case 3:
                        for (n = 0; n < nf; n++) {
                                gx = g + idx[n*3+0];
                                gy = g + idx[n*3+1];
                                gz = g + idx[n*3+2];
                                for (ig = 0; ig < bgrids; ig += SIMDD) {
                                        s = MM_LOAD(gout+ig+bgrids*n);
                                        s += MM_LOAD(gx+ig               ) * MM_LOAD(gy+ig               ) * MM_LOAD(gz+ig               );
                                        s += MM_LOAD(gx+ig+GRID_BLKSIZE  ) * MM_LOAD(gy+ig+GRID_BLKSIZE  ) * MM_LOAD(gz+ig+GRID_BLKSIZE  );
                                        s += MM_LOAD(gx+ig+GRID_BLKSIZE*2) * MM_LOAD(gy+ig+GRID_BLKSIZE*2) * MM_LOAD(gz+ig+GRID_BLKSIZE*2);
                                        MM_STORE(gout+ig+bgrids*n, s);
                                }
                        }
                        break;
                case 4:
                        for (n = 0; n < nf; n++) {
                                gx = g + idx[n*3+0];
                                gy = g + idx[n*3+1];
                                gz = g + idx[n*3+2];
                                for (ig = 0; ig < bgrids; ig += SIMDD) {
                                        s = MM_LOAD(gout+ig+bgrids*n);
                                        s += MM_LOAD(gx+ig               ) * MM_LOAD(gy+ig               ) * MM_LOAD(gz+ig               );
                                        s += MM_LOAD(gx+ig+GRID_BLKSIZE  ) * MM_LOAD(gy+ig+GRID_BLKSIZE  ) * MM_LOAD(gz+ig+GRID_BLKSIZE  );
                                        s += MM_LOAD(gx+ig+GRID_BLKSIZE*2) * MM_LOAD(gy+ig+GRID_BLKSIZE*2) * MM_LOAD(gz+ig+GRID_BLKSIZE*2);
                                        s += MM_LOAD(gx+ig+GRID_BLKSIZE*3) * MM_LOAD(gy+ig+GRID_BLKSIZE*3) * MM_LOAD(gz+ig+GRID_BLKSIZE*3);
                                        MM_STORE(gout+ig+bgrids*n, s);
                                }
                        }
                        break;
                default:
                        for (n = 0; n < nf; n++) {
                                gx = g + idx[n*3+0];
                                gy = g + idx[n*3+1];
                                gz = g + idx[n*3+2];
                                for (ig = 0; ig < bgrids; ig += SIMDD) {
                                        s = MM_LOAD(gout+ig+bgrids*n);
                                        for (i = 0; i < nroots; i++) {
                                                s += MM_LOAD(gx+ig+GRID_BLKSIZE*i) * MM_LOAD(gy+ig+GRID_BLKSIZE*i) * MM_LOAD(gz+ig+GRID_BLKSIZE*i);
                                        }
                                        MM_STORE(gout+ig+bgrids*n, s);
                                }
                        }
                }
        }
}

void CINTnabla1i_grids(double *f, double *g,
                       int li, int lj, CINTEnvVars *envs)
{
        int ngrids = envs->ngrids;
        int bgrids = MIN(ngrids - envs->grids_offset, GRID_BLKSIZE);
        bgrids = ALIGN_UP(bgrids, SIMDD);
        int nroots = envs->nrys_roots;
        const int di = envs->g_stride_i;
        const int dj = envs->g_stride_j;
        int i, j, n, ig, ptr;
        const double *gx = g;
        const double *gy = g + envs->g_size;
        const double *gz = g + envs->g_size * 2;
        double *fx = f;
        double *fy = f + envs->g_size;
        double *fz = f + envs->g_size * 2;
        __MD ai2 = MM_SET1(-2 * envs->ai[0]);

        for (j = 0; j <= lj; j++) {
                //f(...,0,...) = -2*ai*g(...,1,...)
                for (n = 0; n < nroots; n++) {
                        ptr = dj * j + n * GRID_BLKSIZE;
                        for (ig = ptr; ig < ptr+bgrids; ig += SIMDD) {
                                MM_STORE(fx+ig, ai2 * MM_LOAD(gx+ig+di));
                                MM_STORE(fy+ig, ai2 * MM_LOAD(gy+ig+di));
                                MM_STORE(fz+ig, ai2 * MM_LOAD(gz+ig+di));
                        }
                }
                //f(...,i,...) = i*g(...,i-1,...)-2*ai*g(...,i+1,...)
                for (i = 1; i <= li; i++) {
                for (n = 0; n < nroots; n++) {
                        ptr = dj * j + di * i + n * GRID_BLKSIZE;
                        for (ig = ptr; ig < ptr+bgrids; ig += SIMDD) {
                                MM_STORE(fx+ig, MM_SET1(i) * MM_LOAD(gx+ig-di) + ai2 * MM_LOAD(gx+ig+di));
                                MM_STORE(fy+ig, MM_SET1(i) * MM_LOAD(gy+ig-di) + ai2 * MM_LOAD(gy+ig+di));
                                MM_STORE(fz+ig, MM_SET1(i) * MM_LOAD(gz+ig-di) + ai2 * MM_LOAD(gz+ig+di));
                        }
                } }
        }
}

void CINTnabla1j_grids(double *f, double *g,
                       int li, int lj, CINTEnvVars *envs)
{
        int ngrids = envs->ngrids;
        int bgrids = MIN(ngrids - envs->grids_offset, GRID_BLKSIZE);
        bgrids = ALIGN_UP(bgrids, SIMDD);
        int nroots = envs->nrys_roots;
        const int di = envs->g_stride_i;
        const int dj = envs->g_stride_j;
        int i, j, n, ig, ptr;
        const double *gx = g;
        const double *gy = g + envs->g_size;
        const double *gz = g + envs->g_size * 2;
        double *fx = f;
        double *fy = f + envs->g_size;
        double *fz = f + envs->g_size * 2;
        __MD aj2 = MM_SET1(-2 * envs->aj[0]);

        //f(...,0,...) = -2*aj*g(...,1,...)
        for (i = 0; i <= li; i++) {
        for (n = 0; n < nroots; n++) {
                ptr = di * i + n * GRID_BLKSIZE;
                for (ig = ptr; ig < ptr+bgrids; ig += SIMDD) {
                        MM_STORE(fx+ig, aj2 * MM_LOAD(gx+ig+dj));
                        MM_STORE(fy+ig, aj2 * MM_LOAD(gy+ig+dj));
                        MM_STORE(fz+ig, aj2 * MM_LOAD(gz+ig+dj));
                }
        } }
        //f(...,j,...) = j*g(...,j-1,...)-2*aj*g(...,j+1,...)
        for (j = 1; j <= lj; j++) {
        for (i = 0; i <= li; i++) {
        for (n = 0; n < nroots; n++) {
                ptr = dj * j + di * i + n * GRID_BLKSIZE;
                for (ig = ptr; ig < ptr+bgrids; ig += SIMDD) {
                        MM_STORE(fx+ig, MM_SET1(j) * MM_LOAD(gx+ig-dj) + aj2 * MM_LOAD(gx+ig+dj));
                        MM_STORE(fy+ig, MM_SET1(j) * MM_LOAD(gy+ig-dj) + aj2 * MM_LOAD(gy+ig+dj));
                        MM_STORE(fz+ig, MM_SET1(j) * MM_LOAD(gz+ig-dj) + aj2 * MM_LOAD(gz+ig+dj));
                }
        } } }
}


void CINTx1i_grids(double *f, double *g, double *ri,
                   int li, int lj, CINTEnvVars *envs)
{
        int ngrids = envs->ngrids;
        int bgrids = MIN(ngrids - envs->grids_offset, GRID_BLKSIZE);
        bgrids = ALIGN_UP(bgrids, SIMDD);
        int nroots = envs->nrys_roots;
        int i, j, n, ig, ptr;
        const int di = envs->g_stride_i;
        const int dj = envs->g_stride_j;
        const double *gx = g;
        const double *gy = g + envs->g_size;
        const double *gz = g + envs->g_size * 2;
        double *fx = f;
        double *fy = f + envs->g_size;
        double *fz = f + envs->g_size * 2;
        __MD r0 = MM_SET1(ri[0]);
        __MD r1 = MM_SET1(ri[1]);
        __MD r2 = MM_SET1(ri[2]);

        for (j = 0; j <= lj; j++) {
        for (i = 0; i <= li; i++) {
        for (n = 0; n < nroots; n++) {
                ptr = dj * j + di * i + n * GRID_BLKSIZE;
                for (ig = ptr; ig < ptr+bgrids; ig += SIMDD) {
                        MM_STORE(fx+ig, MM_LOAD(gx+ig+di) + r0 * MM_LOAD(gx+ig));
                        MM_STORE(fy+ig, MM_LOAD(gy+ig+di) + r1 * MM_LOAD(gy+ig));
                        MM_STORE(fz+ig, MM_LOAD(gz+ig+di) + r2 * MM_LOAD(gz+ig));
                }
        } } }
}

void CINTx1j_grids(double *f, double *g, double *rj,
                   int li, int lj, CINTEnvVars *envs)
{
        int ngrids = envs->ngrids;
        int bgrids = MIN(ngrids - envs->grids_offset, GRID_BLKSIZE);
        bgrids = ALIGN_UP(bgrids, SIMDD);
        int nroots = envs->nrys_roots;
        int i, j, n, ig, ptr;
        const int di = envs->g_stride_i;
        const int dj = envs->g_stride_j;
        const double *gx = g;
        const double *gy = g + envs->g_size;
        const double *gz = g + envs->g_size * 2;
        double *fx = f;
        double *fy = f + envs->g_size;
        double *fz = f + envs->g_size * 2;
        __MD r0 = MM_SET1(rj[0]);
        __MD r1 = MM_SET1(rj[1]);
        __MD r2 = MM_SET1(rj[2]);

        for (j = 0; j <= lj; j++) {
        for (i = 0; i <= li; i++) {
        for (n = 0; n < nroots; n++) {
                ptr = dj * j + di * i + n * GRID_BLKSIZE;
                for (ig = ptr; ig < ptr+bgrids; ig += SIMDD) {
                        MM_STORE(fx+ig, MM_LOAD(gx+ig+dj) + r0 * MM_LOAD(gx+ig));
                        MM_STORE(fy+ig, MM_LOAD(gy+ig+dj) + r1 * MM_LOAD(gy+ig));
                        MM_STORE(fz+ig, MM_LOAD(gz+ig+dj) + r2 * MM_LOAD(gz+ig));
                }
        } } }
}
