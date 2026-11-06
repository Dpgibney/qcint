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
 * int2e_sph of several shell quartets of one class at once (see cint2e_batch.h).
 *
 *   int2e_sph_batch_ok(shls, bas, env): whether quartet shls can go through int2e_sph_batch: every
 *      shell with l <= 3 and one contracted function, fewer than 8 primitive combinations (int2e_sph
 *      fills its lanes with more), and the full or the long-range Coulomb (env[PTR_RANGE_OMEGA] >= 0).
 *   int2e_sph_batch(out, nq, shls, atm, natm, bas, nbas, env, opt, cache): the nq quartets
 *      shls[4*q..4*q+3], which must share l and the number of primitives of each of their four shells.
 *      out holds nq blocks of (2li+1)(2lj+1)(2lk+1)(2ll+1) doubles in int2e_sph's layout. opt is not
 *      used. cache: int2e_sph_batch_cache_size() doubles, or NULL. Every quartet is computed (no
 *      screening): returns 1, or 0 if nq is 0.
 *   int2e_sph_batch_lanes(): the quartets computed together, SIMDD; more than that are taken in turn.
 */

#include <stdlib.h>
#include <stdint.h>
#include "cint2e_batch.h"

#define EXPORT __attribute__((visibility("default")))

EXPORT int int2e_sph_batch_lanes()
{
        return SIMDD;
}

EXPORT size_t int2e_sph_batch_cache_size()
{
        return (size_t)BATCH_CACHE_BVEC * SIMDD + SIMDD;
}

EXPORT int int2e_sph_batch_ok(int *shls, int *bas, double *env)
{
        if (env[PTR_RANGE_OMEGA] < 0) {
                return 0;
        }
        int ncomb = 1;
        for (int t = 0; t < 4; t++) {
                int sh = shls[t];
                if (bas(ANG_OF, sh) > BATCH_MAXL || bas(NCTR_OF, sh) != 1) {
                        return 0;
                }
                ncomb *= bas(NPRIM_OF, sh);
        }
        return ncomb < 8;
}

static void _batch_gather(CINTBatch2e *b, int nq, int *shls, int *atm, int *bas, double *env)
{
        for (int t = 0; t < 4; t++) {
                b->np[t] = bas(NPRIM_OF, shls[t]);
        }
        for (int lane = 0; lane < SIMDD; lane++) {
                // lanes past the last quartet repeat it
                int *s = shls + 4 * (lane < nq ? lane : nq - 1);
                for (int t = 0; t < 4; t++) {
                        const double *e = env + bas(PTR_EXP, s[t]);
                        const double *c = env + bas(PTR_COEFF, s[t]);
                        for (int p = 0; p < b->np[t]; p++) {
                                b->a[t][p][lane] = e[p];
                                b->c[t][p][lane] = c[p];
                        }
                        const double *r = env + atm(PTR_COORD, bas(ATOM_OF, s[t]));
                        b->r[t][0][lane] = r[0];
                        b->r[t][1][lane] = r[1];
                        b->r[t][2][lane] = r[2];
                }
        }
}

EXPORT CACHE_SIZE_T int2e_sph_batch(double *out, int nq, int *shls, int *atm, int natm,
                                    int *bas, int nbas, double *env, CINTOpt *opt, double *cache)
{
        (void)natm; (void)nbas; (void)opt;
        if (nq <= 0) {
                return 0;
        }
        double *stack = NULL;
        if (cache == NULL) {
                stack = malloc(sizeof(double) * int2e_sph_batch_cache_size());
                cache = stack;
        }
        bvec *work = (bvec *)(((uintptr_t)cache + SIMDD * 8 - 1) & ~(uintptr_t)(SIMDD * 8 - 1));
        const double omega = env[PTR_RANGE_OMEGA];
        const int li = bas(ANG_OF, shls[0]), lj = bas(ANG_OF, shls[1]);
        const int lk = bas(ANG_OF, shls[2]), ll = bas(ANG_OF, shls[3]);
        const size_t nf = (size_t)(2 * li + 1) * (2 * lj + 1) * (2 * lk + 1) * (2 * ll + 1);
        CINTBatch2e b __attribute__((aligned(SIMDD * 8)));
        for (int q0 = 0; q0 < nq; q0 += SIMDD) {
                int n = nq - q0 < SIMDD ? nq - q0 : SIMDD;
                _batch_gather(&b, n, shls + 4 * q0, atm, bas, env);
                double *o = out + nf * q0;
                switch (li) {
                case 0: CINTbatch2e_l0(o, n, &b, omega, work, lj, lk, ll); break;
                case 1: CINTbatch2e_l1(o, n, &b, omega, work, lj, lk, ll); break;
                case 2: CINTbatch2e_l2(o, n, &b, omega, work, lj, lk, ll); break;
                case 3: CINTbatch2e_l3(o, n, &b, omega, work, lj, lk, ll); break;
                }
        }
        if (stack != NULL) {
                free(stack);
        }
        return 1;
}
