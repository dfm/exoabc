# distutils: language = c++
from __future__ import division

cimport cython
from libc.math cimport exp
from libcpp.vector cimport vector
from libcpp.string cimport string

import time
import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

DTYPE_u = np.uint32
ctypedef np.uint32_t DTYPE_u_t


cdef extern from "completeness.h" namespace "exopop":

    cdef cppclass CompletenessModel:
        pass

    cdef cppclass Q1_Q17_CompletenessModel(CompletenessModel):
        Q1_Q17_CompletenessModel (double, double, double, double, double, double)
        double get_pdet (double period, double mes)

    cdef cppclass Star:
        Star ()
        Star (
            CompletenessModel* completeness_model,
            double mass, double radius,
            double dataspan, double dutycycle,
            unsigned n_cdpp, const double* cdpp_x, const double* cdpp_y,
            unsigned n_thresh, const double* thresh_x, const double* thresh_y
        )
        double get_completeness(double q1, double q2, double period, double rp,
                                double incl, double e, double omega)


cdef class DR24DetectionEfficiency:
    """

    """

    def get_pdet(self,
                 np.ndarray[DTYPE_t, ndim=1, mode='c'] params,
                 np.ndarray[DTYPE_t, ndim=1, mode='c'] period,
                 np.ndarray[DTYPE_t, ndim=1, mode='c'] mes):
        # Check the shapes.
        cdef int n = period.shape[0]
        if n != mes.shape[0]:
            raise ValueError("dimension mismatch (period/mes)")

        # Build the completeness model.
        if params.shape[0] != 6:
            raise ValueError("dimension mismatch (params)")
        cdef Q1_Q17_CompletenessModel* completeness_model = \
            new Q1_Q17_CompletenessModel(
                params[0], params[1], params[2], params[3], params[4], params[5],
            )

        cdef int i
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] output = np.empty(n, dtype=DTYPE)
        for i in range(n):
            output[i] = completeness_model.get_pdet(period[i], mes[i])
        del completeness_model
        return output

    def get_detection_efficiency(self, star,
                                 np.ndarray[DTYPE_t, ndim=1, mode='c'] params,
                                 np.ndarray[DTYPE_t, ndim=1, mode='c'] q1s,
                                 np.ndarray[DTYPE_t, ndim=1, mode='c'] q2s,
                                 np.ndarray[DTYPE_t, ndim=1, mode='c'] periods,
                                 np.ndarray[DTYPE_t, ndim=1, mode='c'] radii,
                                 np.ndarray[DTYPE_t, ndim=1, mode='c'] incls,
                                 np.ndarray[DTYPE_t, ndim=1, mode='c'] es,
                                 np.ndarray[DTYPE_t, ndim=1, mode='c'] omegas):
        # Check the shapes.
        cdef int n = q1s.shape[0]
        if (n != q2s.shape[0] or n != periods.shape[0] or n != periods.shape[0]
                or n != radii.shape[0] or n != incls.shape[0]
                or n != es.shape[0] or n != omegas.shape[0]):
            raise ValueError("dimension mismatch")

        # Build the completeness model.
        if params.shape[0] != 6:
            raise ValueError("dimension mismatch (params)")
        cdef Q1_Q17_CompletenessModel* completeness_model = \
            new Q1_Q17_CompletenessModel(
                params[0], params[1], params[2], params[3], params[4], params[5],
            )

        cdef Star* starobj
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] cdpp_x
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] cdpp_y
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] thr_x
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] thr_y
        cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] output = np.empty(n, dtype=DTYPE)

        # Pull out the CDPP values.
        cdpp_cols = [k for k in star.keys() if k.startswith("rrmscdpp")]
        cdpp_x = np.array([k[-4:].replace("p", ".") for k in cdpp_cols],
                            dtype=float)
        inds = np.argsort(cdpp_x)
        cdpp_x = np.ascontiguousarray(cdpp_x[inds], dtype=np.float64)
        cdpp_y = np.ascontiguousarray(star[cdpp_cols][inds],
                                        dtype=np.float64)

        # And the MES thresholds.
        thr_cols = [k for k in star.keys() if k.startswith("mesthres")]
        thr_x = np.array([k[-4:].replace("p", ".") for k in thr_cols],
                            dtype=float)
        inds = np.argsort(thr_x)
        thr_x = np.ascontiguousarray(thr_x[inds], dtype=np.float64)
        thr_y = np.ascontiguousarray(star[thr_cols][inds],
                                        dtype=np.float64)

        # Set up the star.
        starobj = new Star(
            completeness_model,
            star.mass, star.radius, star.dataspan, star.dutycycle,
            cdpp_x.shape[0], <double*>cdpp_x.data, <double*>cdpp_y.data,
            thr_x.shape[0], <double*>thr_x.data, <double*>thr_y.data,
        )

        # Compute the completeness.
        cdef int i
        for i in range(n):
            output[i] = starobj.get_completeness(
                q1s[i], q2s[i], periods[i], radii[i], incls[i], es[i], omegas[i]
            )

        del starobj
        del completeness_model
        return output
