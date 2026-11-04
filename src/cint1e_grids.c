/*
 * Copyright (C) 2021  Qiming Sun <osirpt.sun@gmail.com>
 *
 * <i|1/r|j> integrals for multiple grids
 */

#include <stdlib.h>
#include <math.h>
#include "cint_bas.h"
#include "optimizer.h"
#include "g1e.h"
#include "g1e_grids.h"
#include "misc.h"
#include "cart2sph.h"
#include "c2f.h"
#include "simd.h"

#define PRIM2CTR(ctrsymb, gp, ngp) \
        if (ctrsymb##_ctr > 1) {\
                if (*ctrsymb##empty) { \
                        CINTprim_to_ctr_0(gctr##ctrsymb, gp, c##ctrsymb+ctrsymb##p, \
                                          ngp, ctrsymb##_prim, ctrsymb##_ctr, \
                                          non0ctr##ctrsymb[ctrsymb##p], \
                                          non0idx##ctrsymb+ctrsymb##p*ctrsymb##_ctr); \
                } else { \
                        CINTprim_to_ctr_1(gctr##ctrsymb, gp, c##ctrsymb+ctrsymb##p, \
                                          ngp, ctrsymb##_prim, ctrsymb##_ctr, \
                                          non0ctr##ctrsymb[ctrsymb##p], \
                                          non0idx##ctrsymb+ctrsymb##p*ctrsymb##_ctr); \
                } \
        } \
        *ctrsymb##empty = 0

static void _transpose_comps(double *gctr, double *gctrj,
                             int bgrids, int dij, int ngrids, int n_comp);

int CINT1e_grids_loop(double *gctr, CINTEnvVars *envs, double *cache)
{
        int *shls  = envs->shls;
        int *bas = envs->bas;
        double *env = envs->env;
        int i_sh = shls[0];
        int j_sh = shls[1];
        int i_ctr = envs->x_ctr[0];
        int j_ctr = envs->x_ctr[1];
        int i_prim = bas(NPRIM_OF, i_sh);
        int j_prim = bas(NPRIM_OF, j_sh);
        int nf = envs->nf;
        int n_comp = envs->ncomp_e1 * envs->ncomp_tensor;
        int ngrids = ALIGN_UP(envs->ngrids, SIMDD);
        double *grids = envs->grids;
        double *ai = env + bas(PTR_EXP, i_sh);
        double *aj = env + bas(PTR_EXP, j_sh);
        double *ci = env + bas(PTR_COEFF, i_sh);
        double *cj = env + bas(PTR_COEFF, j_sh);

        double expcutoff = envs->expcutoff;
        PairData *pdata_base, *pdata_ij;
        int *idx;
        int *non0ctri, *non0ctrj;
        int *non0idxi, *non0idxj;
        CINTOpt *opt = envs->opt;
        if (opt != NULL) {
                // the pair data, nonzero coefficients and g indices of int1e_grids_optimizer
                pdata_base = opt->pairdata[i_sh*opt->nbas+j_sh];
                if (pdata_base == NOVALUE) {
                        return 0;
                }
                idx = opt->index_xyz_array[envs->i_l*LMAX1+envs->j_l];
                non0ctri = opt->non0ctr[i_sh];
                non0ctrj = opt->non0ctr[j_sh];
                non0idxi = opt->sortedidx[i_sh];
                non0idxj = opt->sortedidx[j_sh];
        } else {
                double *log_maxci, *log_maxcj;
                MALLOC_INSTACK(log_maxci, i_prim+j_prim);
                MALLOC_INSTACK(pdata_base, i_prim*j_prim);
                log_maxcj = log_maxci + i_prim;
                CINTOpt_log_max_pgto_coeff(log_maxci, ci, i_prim, i_ctr);
                CINTOpt_log_max_pgto_coeff(log_maxcj, cj, j_prim, j_ctr);
                if (CINTset_pairdata(pdata_base, ai, aj, envs->ri, envs->rj,
                                     log_maxci, log_maxcj, envs->li_ceil, envs->lj_ceil,
                                     i_prim, j_prim, SQUARE(envs->rirj), expcutoff, env)) {
                        return 0;
                }
                MALLOC_DATA_INSTACK(idx, nf * 3);
                CINTg2c_index_xyz(idx, envs);
                MALLOC_INSTACK(non0ctri, i_prim+j_prim+i_prim*i_ctr+j_prim*j_ctr);
                non0ctrj = non0ctri + i_prim;
                non0idxi = non0ctrj + j_prim;
                non0idxj = non0idxi + i_prim*i_ctr;
                CINTOpt_non0coeff_byshell(non0idxi, non0ctri, ci, i_prim, i_ctr);
                CINTOpt_non0coeff_byshell(non0idxj, non0ctrj, cj, j_prim, j_ctr);
        }

        double fac1i, fac1j, expij, cutoff;
        double *rij;
        int ip, jp, i, grids_offset, bgrids;
        int empty[4] = {1, 1, 1, 1};
        int *gempty = empty + 0;
        int *iempty = empty + 1;
        int *jempty = empty + 2;
        int all_empty = 1;

        const int nc = i_ctr * j_ctr;
        const int leng = envs->g_size * 3 * ((1<<envs->gbits)+1);
        const int lenj = GRID_BLKSIZE * nf * nc * n_comp; // gctrj
        const int leni = GRID_BLKSIZE * nf * i_ctr * n_comp; // gctri
        const int len0 = GRID_BLKSIZE * nf * n_comp; // gout
        const int len = leng + lenj + leni + len0;
        double *gridsT;
        MALLOC_ALIGNED_DOUBLE_INSTACK(gridsT, len + GRID_BLKSIZE * 3);
        double *g = gridsT + GRID_BLKSIZE * 3;
        double *g1 = g + leng;
        double *gout, *gctri, *gctrj;
        if (n_comp == 1) {
                gctrj = gctr;
        } else {
                gctrj = g1;
                g1 += lenj;
        }
        if (j_ctr == 1) {
                gctri = gctrj;
                iempty = jempty;
        } else {
                gctri = g1;
                g1 += leni;
        }
        if (i_ctr == 1) {
                gout = gctri;
                gempty = iempty;
        } else {
                gout = g1;
        }

        for (grids_offset = 0; grids_offset < ngrids; grids_offset += GRID_BLKSIZE) {
                envs->grids_offset = grids_offset;
                bgrids = MIN(ngrids - grids_offset, GRID_BLKSIZE);
                bgrids = ALIGN_UP(bgrids, SIMDD);
#if (SIMDD == 8) || __AVX2__
                // x, y, z of four points at a time from three loads, by lane permutations and shuffles
                // (cheaper than gathers, which are microcoded on some CPUs)
                for (i = 0; i < bgrids; i += 4) {
                        const double *pg = grids + (grids_offset+i) * 3;
                        __m256d a = _mm256_loadu_pd(pg);      // x0 y0 z0 x1
                        __m256d b = _mm256_loadu_pd(pg + 4);  // y1 z1 x2 y2
                        __m256d c = _mm256_loadu_pd(pg + 8);  // z2 x3 y3 z3
                        __m256d u = _mm256_permute2f128_pd(a, b, 0x30);  // x0 y0 x2 y2
                        __m256d v = _mm256_permute2f128_pd(a, c, 0x21);  // z0 x1 z2 x3
                        __m256d w = _mm256_permute2f128_pd(b, c, 0x30);  // y1 z1 y3 z3
                        _mm256_store_pd(gridsT+i+GRID_BLKSIZE*0, _mm256_shuffle_pd(u, v, 0xa));
                        _mm256_store_pd(gridsT+i+GRID_BLKSIZE*1, _mm256_shuffle_pd(u, w, 0x5));
                        _mm256_store_pd(gridsT+i+GRID_BLKSIZE*2, _mm256_shuffle_pd(v, w, 0xa));
                }
#else
                for (i = 0; i < bgrids; i++) {
                        gridsT[i+GRID_BLKSIZE*0] = grids[(grids_offset+i)*3+0];
                        gridsT[i+GRID_BLKSIZE*1] = grids[(grids_offset+i)*3+1];
                        gridsT[i+GRID_BLKSIZE*2] = grids[(grids_offset+i)*3+2];
                }
#endif

                empty[0] = 1;
                empty[1] = 1;
                empty[2] = 1;
                if (n_comp == 1) {
                        gctrj = gctr + grids_offset * nf * nc;
                        if (j_ctr == 1) {
                                gctri = gctrj;
                        }
                        if (i_ctr == 1) {
                                gout = gctri;
                        }
                }
                pdata_ij = pdata_base;
                for (jp = 0; jp < j_prim; jp++) {
                        envs->aj[0] = aj[jp];
                        if (j_ctr == 1) {
                                fac1j = envs->common_factor * cj[jp];
                        } else {
                                fac1j = envs->common_factor;
                                *iempty = 1;
                        }
                        for (ip = 0; ip < i_prim; ip++, pdata_ij++) {
                                if (pdata_ij->cceij > expcutoff) {
                                        continue;
                                }
                                cutoff = expcutoff - pdata_ij->cceij;
                                envs->ai[0] = ai[ip];
                                expij = pdata_ij->eij;
                                rij = pdata_ij->rij;
                                envs->rij[0] = rij[0];
                                envs->rij[1] = rij[1];
                                envs->rij[2] = rij[2];
                                if (i_ctr == 1) {
                                        fac1i = fac1j*ci[ip]*expij;
                                } else {
                                        fac1i = fac1j*expij;
                                }

                                envs->fac[0] = fac1i;
                                CINTg0_1e_grids(g, cutoff, envs, cache, gridsT);
                                (*envs->f_gout)(gout, g, idx, envs, *gempty);
                                PRIM2CTR(i, gout, bgrids * nf * n_comp);
                        }
                        if (!*iempty) {
                                PRIM2CTR(j, gctri, bgrids * nf * i_ctr * n_comp);
                        }
                }
                if (n_comp > 1 && !*jempty) {
                        _transpose_comps(gctr+grids_offset*nf*nc, gctrj,
                                         bgrids, nf*nc, ngrids, n_comp);
                }
                all_empty &= *jempty;
        }
        return !all_empty;
}

static void _transpose_comps(double *RESTRICT gctr, double *RESTRICT gctrj,
                             int bgrids, int dij, int ngrids, int n_comp)
{
        int n, ic, ig;
        double *pgctr, *pgctrj;
        for (ic = 0; ic < n_comp; ic++) {
                pgctr = gctr + ic * dij * ngrids;
                for (n = 0; n < dij; n++) {
                        pgctrj = gctrj + (n * n_comp + ic) * bgrids;
                        for (ig = 0; ig < bgrids; ig++) {
                                pgctr[ig + n * bgrids] = pgctrj[ig];
                        }
                }
        } 
}

size_t int1e_grids_cache_size(CINTEnvVars *envs)
{
        int *bas = envs->bas;
        int *shls  = envs->shls;
        int *x_ctr = envs->x_ctr;
        int ngrids = ALIGN_UP(envs->ngrids, SIMDD);
        int nroots = envs->nrys_roots;
        int nf = envs->nf;
        int nc = ngrids * nf * x_ctr[0] * x_ctr[1];
        int n_comp = envs->ncomp_e1 * envs->ncomp_tensor;
        int i_prim = bas(NPRIM_OF, shls[0]);
        int j_prim = bas(NPRIM_OF, shls[1]);
        int pdata_size = (i_prim*j_prim * 5
                           + i_prim * x_ctr[0]
                           + j_prim * x_ctr[1]
                           +(i_prim+j_prim)*2 + envs->nf*3);
        size_t leng = envs->g_size*3*((1<<envs->gbits)+1);
        size_t len0 = GRID_BLKSIZE * nf * n_comp;
        size_t leni = len0 * x_ctr[0];
        size_t lenj = leni * x_ctr[1];
        size_t cache_size = MAX(nc*n_comp + leng + len0 + leni + lenj + pdata_size +
                                GRID_BLKSIZE*MAX(n_comp, nroots+6),
                                nc*n_comp + GRID_BLKSIZE * nf*8*OF_CMPLX);
        return cache_size + SIMDD*8;
}

/*
 * 1e integrals <i|O|j> without 1/r
 */
CACHE_SIZE_T CINT1e_grids_drv(double *out, int *dims, CINTEnvVars *envs,
                              double *cache, void (*f_c2s)())
{
        if (out == NULL) {
                return int1e_grids_cache_size(envs);
        }
        int *x_ctr = envs->x_ctr;
        int ngrids_nf = ALIGN_UP(envs->ngrids, SIMDD) * envs->nf;
        int nc = ngrids_nf * x_ctr[0] * x_ctr[1];
        int n_comp = envs->ncomp_e1 * envs->ncomp_tensor;
        double *stack = NULL;
        if (cache == NULL) {
                size_t cache_size = int1e_grids_cache_size(envs);
                stack = malloc(sizeof(double)*cache_size);
                cache = stack;
        }
        double *gctr;
        MALLOC_ALIGNED_DOUBLE_INSTACK(gctr, nc * n_comp);

        int has_value = CINT1e_grids_loop(gctr, envs, cache);

        int counts[4];
        if (dims == NULL) {
                dims = counts;
        }
        if (f_c2s == &c2s_sph_1e_grids) {
                counts[0] = (envs->i_l*2+1) * x_ctr[0];
                counts[1] = (envs->j_l*2+1) * x_ctr[1];
                counts[2] = envs->ngrids;
                counts[3] = 1;
        } else if (f_c2s == &c2s_cart_1e_grids) {
                counts[0] = envs->nfi * x_ctr[0];
                counts[1] = envs->nfj * x_ctr[1];
                counts[2] = envs->ngrids;
                counts[3] = 1;
        }
        int nout = dims[0] * dims[1] * dims[2];
        int n;
        if (has_value) {
                for (n = 0; n < n_comp; n++) {
                        (*f_c2s)(out+nout*n, gctr+nc*n, dims, envs, cache);
                }
        } else {
                for (n = 0; n < n_comp; n++) {
                        c2s_grids_dset0(out+nout*n, dims, counts);
                }
        }
        if (stack != NULL) {
                free(stack);
        }
        return has_value;
}

CACHE_SIZE_T CINT1e_grids_spinor_drv(double complex *out, int *dims, CINTEnvVars *envs,
                                     double *cache, void (*f_c2s)())
{
        if (out == NULL) {
                return int1e_grids_cache_size(envs);
        }
        int *x_ctr = envs->x_ctr;
        int ngrids_nf = ALIGN_UP(envs->ngrids, SIMDD) * envs->nf;
        int nc = ngrids_nf * x_ctr[0] * x_ctr[1] * envs->ncomp_e1;
        double *stack = NULL;
        if (cache == NULL) {
                size_t cache_size = int1e_grids_cache_size(envs);
                stack = malloc(sizeof(double)*cache_size);
                cache = stack;
        }
        double *gctr;
        MALLOC_ALIGNED_DOUBLE_INSTACK(gctr, nc * envs->ncomp_tensor);

        int has_value = CINT1e_grids_loop(gctr, envs, cache);

        int counts[4];
        if (dims == NULL) {
                dims = counts;
        }
        counts[0] = CINTcgto_spinor(envs->shls[0], envs->bas);
        counts[1] = CINTcgto_spinor(envs->shls[1], envs->bas);
        counts[2] = envs->ngrids;
        counts[3] = 1;
        int nout = dims[0] * dims[1] * dims[2];
        int n;
        if (has_value) {
                for (n = 0; n < envs->ncomp_tensor; n++) {
                        (*f_c2s)(out+nout*n, gctr+nc*n, dims, envs, cache);
                }
        } else {
                for (n = 0; n < envs->ncomp_tensor; n++) {
                        c2s_grids_zset0(out+nout*n, dims, counts);
                }
        }
        if (stack != NULL) {
                free(stack);
        }
        return has_value;
}

// the optimizer of int1e_grids_optimizer, if it has the pair data and the call is for omega = 0
static CINTOpt *_grids_opt(CINTOpt *opt, double *env)
{
        if (opt != NULL && opt->pairdata != NULL && env[PTR_RANGE_OMEGA] == 0) {
                return opt;
        }
        return NULL;
}

CACHE_SIZE_T int1e_grids_sph(double *out, int *dims, int *shls, int *atm, int natm,
                     int *bas, int nbas, double *env, CINTOpt *opt, double *cache)
{
        int ng[] = {0, 0, 0, 0, 0, 1, 0, 1};
        CINTEnvVars envs;
        CINTinit_int1e_grids_EnvVars(&envs, ng, shls, atm, natm, bas, nbas, env);
        envs.f_gout = &CINTgout1e_grids;
        envs.opt = _grids_opt(opt, env);
        return CINT1e_grids_drv(out, dims, &envs, cache, &c2s_sph_1e_grids);
}

/*
 * The pair data of every ordered pair of shells, made as CINT1e_grids_loop makes them per call (CINTset_pairdata
 * with the same arguments), so that the integrals are bitwise the same with the optimizer as without it; NOVALUE
 * for a pair whose primitive pairs are all screened. As in CINTOpt_setij, pairdata[0] is the start of the buffer.
 */
static void _grids_pair(CINTOpt *opt, PairData *pdata, int i, int j, double expcutoff,
                        int *atm, int *bas, double *env, int *empty)
{
        double *ri = env + atm(PTR_COORD, bas(ATOM_OF, i));
        double *rj = env + atm(PTR_COORD, bas(ATOM_OF, j));
        int li = bas(ANG_OF, i);
        int lj = bas(ANG_OF, j);
        double rirj[3];
        int k;
        // the orientation of CINTinit_int1e_grids_EnvVars
        for (k = 0; k < 3; k++) {
                rirj[k] = li > lj ? ri[k] - rj[k] : rj[k] - ri[k];
        }
        *empty = CINTset_pairdata(pdata, env + bas(PTR_EXP, i), env + bas(PTR_EXP, j), ri, rj,
                                  opt->log_max_coeff[i], opt->log_max_coeff[j], li, lj,
                                  bas(NPRIM_OF, i), bas(NPRIM_OF, j), SQUARE(rirj), expcutoff, env);
}

static void _grids_setij(CINTOpt *opt, int *atm, int natm, int *bas, int nbas, double *env)
{
        size_t tot_prim = 0;
        int i, j, empty;
        for (i = 0; i < nbas; i++) {
                tot_prim += bas(NPRIM_OF, i);
        }
        if (tot_prim == 0 || tot_prim > MAX_PGTO_FOR_PAIRDATA) {
                return;
        }
        double expcutoff;
        if (env[PTR_EXPCUTOFF] == 0) {
                expcutoff = EXPCUTOFF;
        } else {
                expcutoff = MAX(MIN_EXPCUTOFF, env[PTR_EXPCUTOFF]);
        }
        opt->pairdata = malloc(sizeof(PairData *) * MAX(nbas * nbas, 1));
        PairData *pdata = malloc(sizeof(PairData) * tot_prim * tot_prim);
        opt->pairdata[0] = pdata;
        // (i, j) and then (j, i) next to it, the layout of CINTOpt_setij; each made by its own call
        for (i = 0; i < nbas; i++) {
                int iprim = bas(NPRIM_OF, i);
                for (j = 0; j <= i; j++) {
                        int jprim = bas(NPRIM_OF, j);
                        _grids_pair(opt, pdata, i, j, expcutoff, atm, bas, env, &empty);
                        if (i == 0 && j == 0) {
                                // kept even if empty: the loop then skips all its primitive pairs
                                pdata += iprim * jprim;
                                continue;
                        }
                        if (empty) {
                                opt->pairdata[i*nbas+j] = NOVALUE;
                        } else {
                                opt->pairdata[i*nbas+j] = pdata;
                                pdata += iprim * jprim;
                        }
                        if (i != j) {
                                _grids_pair(opt, pdata, j, i, expcutoff, atm, bas, env, &empty);
                                if (empty) {
                                        opt->pairdata[j*nbas+i] = NOVALUE;
                                } else {
                                        opt->pairdata[j*nbas+i] = pdata;
                                        pdata += iprim * jprim;
                                }
                        }
                }
        }
}

/*
 * The pair data (exponents, screening and centers of the primitive pairs), the nonzero coefficients and the g
 * indices of all shells, which CINT1e_grids_loop would otherwise make on every call. The pair data and the g
 * indices depend on the range-separation parameter (its screening estimate; omega < 0 doubles the roots), so the
 * optimizer is made for omega = 0 only and used only when the call's omega is 0 (_grids_opt).
 */
void int1e_grids_optimizer(CINTOpt **opt, int *atm, int natm,
                           int *bas, int nbas, double *env)
{
        if (env[PTR_RANGE_OMEGA] != 0) {
                *opt = NULL;
                return;
        }
        int ng[] = {0, 0, 0, 0, 0, 1, 0, 1};
        CINTall_1e_grids_optimizer(opt, ng, atm, natm, bas, nbas, env);
        _grids_setij(*opt, atm, natm, bas, nbas, env);
}


CACHE_SIZE_T int1e_grids_cart(double *out, int *dims, int *shls, int *atm, int natm,
                              int *bas, int nbas, double *env, CINTOpt *opt, double *cache)
{
        int ng[] = {0, 0, 0, 0, 0, 1, 0, 1};
        CINTEnvVars envs;
        CINTinit_int1e_grids_EnvVars(&envs, ng, shls, atm, natm, bas, nbas, env);
        envs.f_gout = &CINTgout1e_grids;
        envs.opt = _grids_opt(opt, env);
        return CINT1e_grids_drv(out, dims, &envs, cache, &c2s_cart_1e_grids);
}

CACHE_SIZE_T int1e_grids_spinor(double complex *out, int *dims, int *shls, int *atm, int natm,
                                int *bas, int nbas, double *env, CINTOpt *opt, double *cache)
{
        int ng[] = {0, 0, 0, 0, 0, 1, 0, 1};
        CINTEnvVars envs;
        CINTinit_int1e_grids_EnvVars(&envs, ng, shls, atm, natm, bas, nbas, env);
        envs.f_gout = &CINTgout1e_grids;
        envs.opt = _grids_opt(opt, env);
        return CINT1e_grids_spinor_drv(out, dims, &envs, cache, &c2s_sf_1e_grids);
}

ALL_CINT(int1e_grids)

