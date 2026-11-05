/*
 * int1e_grids integrals contracted as they are made, for PCM's Gaussian surface charges, as GPU4PySCF's int3c1e
 * kernels do it:
 *
 *   int1e_grids_dm_contract_cart:     v[k*nv+g] += sum_ij dm[k*nij+i+di*j] (ij|g)
 *   int1e_grids_charge_contract_cart: vij[k*nij+i+di*j] += sum_g q[k*nv+g] (ij|g)
 *
 * for the Cartesian functions of the shell pair shls[0], shls[1] (di and dj of them, nij = di*dj, k < n_dm) and
 * the points of env[PTR_GRIDS], in batches: batch b holds the points [batch_loc[b], batch_loc[b+1]) and takes
 * env[PTR_RINV_ZETA] = batch_zeta[b], so env is changed. With v (or vij) NULL they return the size of the cache
 * they need, in doubles.
 *
 * They take only the vertical recursion, for li+lj on the center A of the larger shell: since
 * (x-B)^j = sum_k binom(j,k) (A-B)^(j-k) (x-A)^k, the density is moved once to the monomials t = i+k on A, and
 * each point takes one dot product of those with the g arrays of the vertical recursion. For the charges, the
 * sums over the points are kept for the monomials, and the horizontal recursion is applied to them at the end.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cint_bas.h"
#include "optimizer.h"
#include "g1e.h"
#include "g1e_grids.h"
#include "misc.h"
#include "simd.h"

#define CONTRACT_DM     0
#define CONTRACT_CHARGE 1

// the Cartesian components (lx, ly, lz) of l, in libcint's order
static void _cart_comps(int *lx, int *ly, int *lz, int l)
{
        int n = 0;
        int x, y;
        for (x = l; x >= 0; x--) {
                for (y = l - x; y >= 0; y--, n++) {
                        lx[n] = x;
                        ly[n] = y;
                        lz[n] = l - x - y;
                }
        }
}

// the monomials t of li <= tx+ty+tz <= lij in the order of the dot products: tidx[(tx*l1+ty)*l1+tz] is the place
// of t, -1 for those not taken; returns their number
static int _monomials(int *tidx, int li, int lij)
{
        int l1 = lij + 1;
        int n = 0;
        int i, tx, ty, tz;
        for (i = 0; i < l1*l1*l1; i++) {
                tidx[i] = -1;
        }
        for (tx = 0; tx <= lij; tx++) {
        for (ty = 0; ty <= lij - tx; ty++) {
                for (tz = MAX(li - tx - ty, 0); tz <= lij - tx - ty; tz++, n++) {
                        tidx[(tx*l1+ty)*l1+tz] = n;
                }
        } }
        return n;
}

// h[j*(lj+1)+k] = binom(j,k) ab^(j-k), k <= j <= lj
static void _hrr_coeff(double *h, double ab, int lj)
{
        int n = lj + 1;
        int j, k;
        for (j = 0; j <= lj; j++) {
                double binom = 1;
                double p = 1;
                for (k = j; k >= 0; k--) {
                        h[j*n+k] = binom * p;
                        // binom(j,k-1) = binom(j,k) k / (j-k+1)
                        binom = binom * k / (j - k + 1);
                        p *= ab;
                }
        }
}

// out[t] += c[a,b] prod_d h_d[b_d, t_d-a_d] over the Cartesian pairs (a, b) of la, lb (or, with transpose, c[a,b]
// += prod ... out[t]): the horizontal recursion between the functions of A and B and the monomials of A
static void _hrr_monomials(double *t_coeff, double *ab_block, int lda, int transpose,
                           int la, int lb, int *ax, int *ay, int *az, int *bx, int *by, int *bz,
                           double *hx, double *hy, double *hz, int *tidx, int l1)
{
        int nfa = (la + 1) * (la + 2) / 2;
        int nfb = (lb + 1) * (lb + 2) / 2;
        int n = lb + 1;
        int a, b, kx, ky, kz;
        for (b = 0; b < nfb; b++) {
        for (a = 0; a < nfa; a++) {
                double s = 0;
                double c = transpose ? 0 : ab_block[a+lda*b];
                for (kx = 0; kx <= bx[b]; kx++) {
                        double cx = hx[bx[b]*n+kx];
                        int tx = ax[a] + kx;
                        for (ky = 0; ky <= by[b]; ky++) {
                                double cxy = cx * hy[by[b]*n+ky];
                                int ty = ay[a] + ky;
                                int *it = tidx + (tx*l1+ty)*l1 + az[a];
                                double *hzb = hz + bz[b]*n;
                                for (kz = 0; kz <= bz[b]; kz++) {
                                        if (transpose) {
                                                s += cxy * hzb[kz] * t_coeff[it[kz]];
                                        } else {
                                                t_coeff[it[kz]] += c * cxy * hzb[kz];
                                        }
                                }
                        }
                }
                if (transpose) {
                        ab_block[a+lda*b] += s;
                }
        } }
}

static CACHE_SIZE_T _grids_contract(double *out, double *in, int n_dm, int nv, int *shls,
                                    int *atm, int natm, int *bas, int nbas, double *env,
                                    CINTOpt *opt, double *cache, int nbatch, int *batch_loc,
                                    double *batch_zeta, int mode)
{
        // A, the shell of the larger l, is the base of the vertical recursion
        int swap = bas(ANG_OF, shls[1]) > bas(ANG_OF, shls[0]);
        int shls_ab[4] = {shls[swap], shls[1-swap], 0, 0};
        int ng[] = {0, 0, 0, 0, 0, 1, 0, 1};
        CINTEnvVars envs;
        CINTinit_int1e_grids_EnvVars(&envs, ng, shls_ab, atm, natm, bas, nbas, env);
        int i_sh = shls_ab[0];
        int j_sh = shls_ab[1];
        int li = envs.i_l;
        int lj = envs.j_l;
        int lij = li + lj;
        int l1 = lij + 1;
        int nfi = envs.nfi;
        int nfj = envs.nfj;
        int i_ctr = envs.x_ctr[0];
        int j_ctr = envs.x_ctr[1];
        int nc = i_ctr * j_ctr;
        int di = nfi * i_ctr;
        int dj = nfj * j_ctr;
        int nij = di * dj;
        int i_prim = bas(NPRIM_OF, i_sh);
        int j_prim = bas(NPRIM_OF, j_sh);
        int nroots = envs.nrys_roots;
        // the numbers of monomials of degrees <= lij and < li
        int nt = l1 * (l1+1) * (l1+2) / 6 - li * (li+1) * (li+2) / 6;
        // the vertical recursion only, li+lj on A, nothing on B
        int li_ceil = envs.li_ceil;
        int lj_ceil = envs.lj_ceil;
        envs.li_ceil = lij;
        envs.lj_ceil = 0;
        envs.g_stride_i = GRID_BLKSIZE * nroots;
        envs.g_stride_j = GRID_BLKSIZE * nroots * l1;
        envs.g_size = GRID_BLKSIZE * nroots * l1;
        const int dg = envs.g_stride_i;

        size_t cache_size = 3 * (size_t)envs.g_size + GRID_BLKSIZE * (nroots + 6)
                + (size_t)n_dm * nc * nt + (size_t)n_dm * nt + (size_t)nt * SIMDD
                + (size_t)n_dm * GRID_BLKSIZE + 3 * (lj+1) * (lj+1) + nc + nfi * nfj
                + l1*l1*l1 + 3 * (nfi+nfj) + i_prim*j_prim*5 + i_prim+j_prim + SIMDD*16;
        if (out == NULL) {
                return cache_size;
        }
        double *stack = NULL;
        if (cache == NULL) {
                stack = malloc(sizeof(double) * cache_size);
                cache = stack;
        }

        double *ai = env + bas(PTR_EXP, i_sh);
        double *aj = env + bas(PTR_EXP, j_sh);
        double *ci = env + bas(PTR_COEFF, i_sh);
        double *cj = env + bas(PTR_COEFF, j_sh);
        double expcutoff = envs.expcutoff;
        CINTOpt *gopt = NULL;
        if (opt != NULL && opt->pairdata != NULL && env[PTR_RANGE_OMEGA] == 0) {
                gopt = opt;
        }
        PairData *pdata_base, *pdata_ij;
        if (gopt != NULL) {
                // the pair data of int1e_grids_optimizer, as CINT1e_grids_loop takes them
                pdata_base = gopt->pairdata[i_sh*gopt->nbas+j_sh];
                if (pdata_base == NOVALUE) {
                        if (stack != NULL) {
                                free(stack);
                        }
                        return 0;
                }
        } else {
                double *log_maxci, *log_maxcj;
                MALLOC_INSTACK(log_maxci, i_prim+j_prim);
                MALLOC_INSTACK(pdata_base, i_prim*j_prim);
                log_maxcj = log_maxci + i_prim;
                CINTOpt_log_max_pgto_coeff(log_maxci, ci, i_prim, i_ctr);
                CINTOpt_log_max_pgto_coeff(log_maxcj, cj, j_prim, j_ctr);
                if (CINTset_pairdata(pdata_base, ai, aj, envs.ri, envs.rj,
                                     log_maxci, log_maxcj, li_ceil, lj_ceil,
                                     i_prim, j_prim, SQUARE(envs.rirj), expcutoff, env)) {
                        if (stack != NULL) {
                                free(stack);
                        }
                        return 0;
                }
        }

        double *g, *gridsT, *dt, *dab, *sacc, *buf, *hx, *hy, *hz, *cw, *blk;
        int *tidx, *ix, *iy, *iz, *jx, *jy, *jz;
        MALLOC_INSTACK(g, envs.g_size * 3);
        MALLOC_INSTACK(gridsT, GRID_BLKSIZE * 3);
        MALLOC_INSTACK(dt, n_dm * nc * nt);
        MALLOC_INSTACK(dab, n_dm * nt);
        MALLOC_INSTACK(sacc, nt * SIMDD);
        MALLOC_INSTACK(buf, n_dm * GRID_BLKSIZE);
        MALLOC_INSTACK(hx, (lj+1) * (lj+1) * 3);
        MALLOC_INSTACK(cw, nc);
        // one (nfi, nfj) block of the density or of the result
        MALLOC_INSTACK(blk, nfi * nfj);
        hy = hx + (lj+1) * (lj+1);
        hz = hy + (lj+1) * (lj+1);
        MALLOC_DATA_INSTACK(tidx, l1*l1*l1 + 3 * (nfi+nfj));
        ix = tidx + l1*l1*l1;
        iy = ix + nfi;
        iz = iy + nfi;
        jx = iz + nfi;
        jy = jx + nfj;
        jz = jy + nfj;
        double *gcache = cache;
        _monomials(tidx, li, lij);
        _cart_comps(ix, iy, iz, li);
        _cart_comps(jx, jy, jz, lj);
        _hrr_coeff(hx, envs.ri[0] - envs.rj[0], lj);
        _hrr_coeff(hy, envs.ri[1] - envs.rj[1], lj);
        _hrr_coeff(hz, envs.ri[2] - envs.rj[2], lj);

        int k, ic, jc, ip, jp, i, n, b, p0, ig, tx, ty, tz, it;
        // the caller's blocks are in (shls[0], shls[1]) order, element (a, b) at a + rows*b
        int stride_a = swap ? dj : 1;
        int stride_b = swap ? 1 : di;
        memset(dt, 0, sizeof(double) * n_dm * nc * nt);
        if (mode == CONTRACT_DM) {
                // the density on the monomials of A, for each pair of contracted functions
                for (k = 0; k < n_dm; k++) {
                for (jc = 0; jc < j_ctr; jc++) {
                for (ic = 0; ic < i_ctr; ic++) {
                        for (int bj = 0; bj < nfj; bj++) {
                        for (int ai_ = 0; ai_ < nfi; ai_++) {
                                int alpha = ic * nfi + ai_;
                                int beta = jc * nfj + bj;
                                blk[ai_+nfi*bj] = in[k*nij + alpha*stride_a + beta*stride_b];
                        } }
                        _hrr_monomials(dt + ((k*i_ctr+ic)*j_ctr+jc)*nt, blk, nfi, 0,
                                       li, lj, ix, iy, iz, jx, jy, jz, hx, hy, hz, tidx, l1);
                } } }
        }

        double fac1j;
        double *grids = env + (size_t)env[PTR_GRIDS];
        double *gx = g;
        double *gy = g + envs.g_size;
        double *gz = g + envs.g_size * 2;
        int has_value = 0;
        for (b = 0; b < nbatch; b++) {
                env[PTR_RINV_ZETA] = batch_zeta[b];
                for (p0 = batch_loc[b]; p0 < batch_loc[b+1]; p0 += GRID_BLKSIZE) {
                        int nb = MIN(batch_loc[b+1] - p0, GRID_BLKSIZE);
                        int nbp = ALIGN_UP(nb, SIMDD);
                        // the block's points, transposed; the padding repeats the last point
                        for (i = 0; i < nbp; i++) {
                                int p = p0 + MIN(i, nb - 1);
                                gridsT[i+GRID_BLKSIZE*0] = grids[p*3+0];
                                gridsT[i+GRID_BLKSIZE*1] = grids[p*3+1];
                                gridsT[i+GRID_BLKSIZE*2] = grids[p*3+2];
                        }
                        envs.ngrids = nbp;
                        envs.grids_offset = 0;
                        if (mode == CONTRACT_DM) {
                                memset(buf, 0, sizeof(double) * n_dm * GRID_BLKSIZE);
                        } else {
                                for (k = 0; k < n_dm; k++) {
                                        for (i = 0; i < nb; i++) {
                                                buf[k*GRID_BLKSIZE+i] = in[k*nv+p0+i];
                                        }
                                        for (; i < nbp; i++) {
                                                buf[k*GRID_BLKSIZE+i] = 0;
                                        }
                                }
                        }

                        pdata_ij = pdata_base;
                        for (jp = 0; jp < j_prim; jp++) {
                                envs.aj[0] = aj[jp];
                                if (j_ctr == 1) {
                                        fac1j = envs.common_factor * cj[jp];
                                } else {
                                        fac1j = envs.common_factor;
                                }
                                for (ip = 0; ip < i_prim; ip++, pdata_ij++) {
                                        if (pdata_ij->cceij > expcutoff) {
                                                continue;
                                        }
                                        has_value = 1;
                                        double cutoff = expcutoff - pdata_ij->cceij;
                                        envs.ai[0] = ai[ip];
                                        envs.rij[0] = pdata_ij->rij[0];
                                        envs.rij[1] = pdata_ij->rij[1];
                                        envs.rij[2] = pdata_ij->rij[2];
                                        if (i_ctr == 1) {
                                                envs.fac[0] = fac1j * ci[ip] * pdata_ij->eij;
                                        } else {
                                                envs.fac[0] = fac1j * pdata_ij->eij;
                                        }
                                        CINTg0_1e_grids(g, cutoff, &envs, gcache, gridsT);

                                        // the contraction coefficients of this primitive pair not in fac
                                        for (jc = 0; jc < j_ctr; jc++) {
                                        for (ic = 0; ic < i_ctr; ic++) {
                                                cw[ic*j_ctr+jc] = (i_ctr == 1 ? 1. : ci[ip+i_prim*ic])
                                                                * (j_ctr == 1 ? 1. : cj[jp+j_prim*jc]);
                                        } }

                                        if (mode == CONTRACT_DM) {
                                                double *d;
                                                for (k = 0; k < n_dm; k++) {
                                                        if (nc == 1) {
                                                                d = dt + k * nt;
                                                        } else {
                                                                d = dab + k * nt;
                                                                memset(d, 0, sizeof(double) * nt);
                                                                for (n = 0; n < nc; n++) {
                                                                        double c = cw[n];
                                                                        double *dtn = dt + (k*nc+n)*nt;
                                                                        for (it = 0; it < nt; it++) {
                                                                                d[it] += c * dtn[it];
                                                                        }
                                                                }
                                                        }
                                                        double *vb = buf + k * GRID_BLKSIZE;
                                                        for (ig = 0; ig < nbp; ig += SIMDD) {
                                                                __MD acc = MM_SET1(0.);
                                                                for (n = 0; n < nroots; n++) {
                                                                        double *pgx = gx + GRID_BLKSIZE*n + ig;
                                                                        double *pgy = gy + GRID_BLKSIZE*n + ig;
                                                                        double *pgz = gz + GRID_BLKSIZE*n + ig;
                                                                        it = 0;
                                                                        for (tx = 0; tx <= lij; tx++) {
                                                                                __MD accx = MM_SET1(0.);
                                                                                for (ty = 0; ty <= lij - tx; ty++) {
                                                                                        __MD accz = MM_SET1(0.);
                                                                                        for (tz = MAX(li-tx-ty, 0); tz <= lij-tx-ty; tz++, it++) {
                                                                                                accz = MM_FMA(MM_SET1(d[it]), MM_LOAD(pgz+tz*dg), accz);
                                                                                        }
                                                                                        accx = MM_FMA(MM_LOAD(pgy+ty*dg), accz, accx);
                                                                                }
                                                                                acc = MM_FMA(MM_LOAD(pgx+tx*dg), accx, acc);
                                                                        }
                                                                }
                                                                MM_STORE(vb+ig, MM_ADD(MM_LOAD(vb+ig), acc));
                                                        }
                                                }
                                        } else {
                                                for (k = 0; k < n_dm; k++) {
                                                        double *qb = buf + k * GRID_BLKSIZE;
                                                        memset(sacc, 0, sizeof(double) * nt * SIMDD);
                                                        for (ig = 0; ig < nbp; ig += SIMDD) {
                                                                __MD rq = MM_LOAD(qb+ig);
                                                                for (n = 0; n < nroots; n++) {
                                                                        double *pgx = gx + GRID_BLKSIZE*n + ig;
                                                                        double *pgy = gy + GRID_BLKSIZE*n + ig;
                                                                        double *pgz = gz + GRID_BLKSIZE*n + ig;
                                                                        it = 0;
                                                                        for (tx = 0; tx <= lij; tx++) {
                                                                                __MD qx = MM_MUL(rq, MM_LOAD(pgx+tx*dg));
                                                                                for (ty = 0; ty <= lij - tx; ty++) {
                                                                                        __MD qxy = MM_MUL(qx, MM_LOAD(pgy+ty*dg));
                                                                                        for (tz = MAX(li-tx-ty, 0); tz <= lij-tx-ty; tz++, it++) {
                                                                                                MM_STORE(sacc+it*SIMDD, MM_FMA(qxy, MM_LOAD(pgz+tz*dg),
                                                                                                                               MM_LOAD(sacc+it*SIMDD)));
                                                                                        }
                                                                                }
                                                                        }
                                                                }
                                                        }
                                                        for (it = 0; it < nt; it++) {
                                                                double s = 0;
                                                                for (i = 0; i < SIMDD; i++) {
                                                                        s += sacc[it*SIMDD+i];
                                                                }
                                                                for (n = 0; n < nc; n++) {
                                                                        dt[(k*nc+n)*nt+it] += cw[n] * s;
                                                                }
                                                        }
                                                }
                                        }
                                }
                        }
                        if (mode == CONTRACT_DM) {
                                for (k = 0; k < n_dm; k++) {
                                        for (i = 0; i < nb; i++) {
                                                out[k*nv+p0+i] += buf[k*GRID_BLKSIZE+i];
                                        }
                                }
                        }
                }
        }

        if (mode == CONTRACT_CHARGE && has_value) {
                // the sums on the monomials of A back to the functions of A and B
                for (k = 0; k < n_dm; k++) {
                for (jc = 0; jc < j_ctr; jc++) {
                for (ic = 0; ic < i_ctr; ic++) {
                        memset(blk, 0, sizeof(double) * nfi * nfj);
                        _hrr_monomials(dt + ((k*i_ctr+ic)*j_ctr+jc)*nt, blk, nfi, 1,
                                       li, lj, ix, iy, iz, jx, jy, jz, hx, hy, hz, tidx, l1);
                        for (int bj = 0; bj < nfj; bj++) {
                        for (int ai_ = 0; ai_ < nfi; ai_++) {
                                int alpha = ic * nfi + ai_;
                                int beta = jc * nfj + bj;
                                out[k*nij + alpha*stride_a + beta*stride_b] += blk[ai_+nfi*bj];
                        } }
                } } }
        }
        if (stack != NULL) {
                free(stack);
        }
        return has_value;
}

CACHE_SIZE_T int1e_grids_dm_contract_cart(double *v, double *dm, int n_dm, int nv, int *shls,
                                          int *atm, int natm, int *bas, int nbas, double *env,
                                          CINTOpt *opt, double *cache,
                                          int nbatch, int *batch_loc, double *batch_zeta)
{
        return _grids_contract(v, dm, n_dm, nv, shls, atm, natm, bas, nbas, env, opt, cache,
                               nbatch, batch_loc, batch_zeta, CONTRACT_DM);
}

CACHE_SIZE_T int1e_grids_charge_contract_cart(double *vij, double *q, int n_dm, int nv, int *shls,
                                              int *atm, int natm, int *bas, int nbas, double *env,
                                              CINTOpt *opt, double *cache,
                                              int nbatch, int *batch_loc, double *batch_zeta)
{
        return _grids_contract(vij, q, n_dm, nv, shls, atm, natm, bas, nbas, env, opt, cache,
                               nbatch, batch_loc, batch_zeta, CONTRACT_CHARGE);
}
