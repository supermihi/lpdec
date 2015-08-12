# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

"""cdef headers for talking to the GLPK C library through Cython. See the GLPK manual for
explanations of the interface."""

cdef extern from 'glpk.h':
    ctypedef struct glp_prob:
        pass
    ctypedef struct glp_smcp:
        int msg_lev
        int meth
        int presolve
        double tol_bnd
        double tol_dj
        double obj_ul
    ctypedef struct glp_cpxcp:
        pass

    const int GLP_MIN, GLP_MAX
    const int GLP_UP, GLP_LO, GLP_DB, GLP_FX
    const int GLP_MSG_OFF, GLP_MSG_ERR, GLP_MSG_ALL
    const int GLP_DUAL, GLP_PRIMAL
    const int GLP_OPT, GLP_NOFEAS, GLP_UNBND, GLP_EOBJUL
    const int GLP_BS, GLP_NL, GLP_NU
    const int GLP_ON, GLP_OFF
    
    glp_prob *glp_create_prob()
    void glp_erase_prob(glp_prob *P)
    void glp_set_obj_dir(glp_prob *P, int dir)
    int glp_add_rows(glp_prob *P, int nrs)
    int glp_add_cols(glp_prob *P, int ncs)
    void glp_set_row_bnds(glp_prob *P, int i, int type, double lb, double ub)
    void glp_set_col_bnds(glp_prob *P, int j, int type, double lb, double ub)
    void glp_set_col_bnds(glp_prob *P, int j, int type, double lb, double ub)
    void glp_set_obj_coef(glp_prob *P, int j, double coef)
    void glp_set_mat_row(glp_prob *P, int i, int len, const int ind[], const double val[])
    void glp_del_rows(glp_prob *P, int nrs, const int num[])
    int glp_get_num_rows(glp_prob *P)
    int glp_simplex(glp_prob *P, const glp_smcp *parm)
    int glp_get_status(glp_prob *P)
    int glp_init_smcp(glp_smcp *parm)
    double glp_get_obj_val(glp_prob *P)
    double glp_get_col_prim(glp_prob *P, int j)
    double glp_get_row_prim(glp_prob *P, int i)
    void glp_std_basis(glp_prob *P)
    void glp_adv_basis(glp_prob *P, int flags)
    void glp_cpx_basis(glp_prob *P)
    int glp_get_row_stat(glp_prob *P, int i)
    int glp_get_mat_row(glp_prob *P, int i, int ind[], double val[])
    double glp_get_row_ub(glp_prob *P, int i)
    int glp_write_lp(glp_prob *P, const glp_cpxcp *parm, const char *fname)