# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

"""cdef headers for talking to the Gurobi C library through Cython."""

cdef extern from "gurobi_c.h":
    ctypedef struct GRBenv:
        pass
    ctypedef struct GRBmodel:
        pass
        
    const char GRB_BINARY, GRB_CONTINUOUS, GRB_INTEGER
    const char GRB_EQUAL, GRB_LESS_EQUAL, GRB_GREATER_EQUAL
    const char *GRB_INT_ATTR_MODELSENSE
    const char *GRB_DBL_ATTR_OBJ
    const char *GRB_DBL_ATTR_X
    const char *GRB_DBL_ATTR_OBJVAL
    const char *GRB_INT_PAR_OUTPUTFLAG
    const char *GRB_INT_ATTR_NUMCONSTRS
    const char *GRB_INT_ATTR_STATUS
    const char *GRB_DBL_ATTR_ITERCOUNT
    const char *GRB_DBL_ATTR_SLACK
    const char *GRB_DBL_ATTR_LB
    const char *GRB_DBL_ATTR_UB
    const char *GRB_INT_PAR_METHOD
    const int GRB_MAXIMIZE, GRB_MINIMIZE, GRB_INFEASIBLE, GRB_OPTIMAL, GRB_INTERRUPTED, GRB_INF_OR_UNBD
    const double GRB_INFINITY
        
    int GRBloadenv(GRBenv **envP, const char *logfilename)
    int GRBnewmodel (GRBenv *env, GRBmodel **modelP, const char *Pname, int numvars, double *obj,
                     double *lb, double *ub, char *vtype, const char **varnames )
    int GRBresetmodel (GRBmodel *model)
    int GRBfreemodel (GRBmodel *model)
    int GRBaddvar (GRBmodel *model, int numnz, int *vind, double *vval, double obj, double lb,
                   double ub, char vtype, const char *varname )
    int GRBsetintattr (GRBmodel *model, const char *attrname, int newvalue)
    int GRBgetintattr (GRBmodel *model, const char *attrname, int *valueP)
    int GRBgetdblattr (GRBmodel *model, const char *attrname, double *valueP)
    int GRBsetdblattrelement (GRBmodel *model, const char *attrname, int element, double newvalue)
    int GRBgetdblattrelement (GRBmodel *model, const char *attrname, int element, double *valueP)
    int GRBsetdblattrarray (GRBmodel *model, const char *attrname, int start, int len, double *values)
    int GRBgetdblattrarray (GRBmodel *model, const char *attrname, int start, int len, double *values)
    int GRBsetintparam (GRBenv *env, const char *paramname, int newvalue)
    int GRBupdatemodel (GRBmodel *model)
    int GRBaddconstr (GRBmodel *model, int numnz, int *cind, double *cval, char sense, double rhs, const char *constrname)
    int GRBdelconstrs (GRBmodel *model, int numdel, int *ind)
    int GRBoptimize (GRBmodel *model)
    int GRBwrite(GRBmodel *model, const char *filename)