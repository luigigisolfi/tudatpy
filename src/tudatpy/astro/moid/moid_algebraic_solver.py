import numpy as np


class AlgebraicMOIDSolver:
    """
    Algebraic MOID (Minimum Orbit Intersection Distance) solver.

    The minimal orbit intersection distance (MOID) is a measure for the distance
    between the orbits of a NEO and of the Earth,
    not considering the positions that the bodies occupy in them.
    → can act as early warning indicator for collisions, for tasking of follow-up observations and analysis.

    NOTE: The MOID is computed using UNPERTURBED (Keplerian) orbits.

    Implements the resultant-based method of Kholshevnikov & Vassiliev (1999) /
    Gronchi (2002, 2005), following the explicit formulas given in:

        Grassi, Gronchi & Bau (2023), "Revisiting the computation of the
        critical points of the Keplerian distance", arXiv:2305.13900,
        Section 3 ("Eccentric anomalies and ordinary polynomials").

    Pipeline:
      1. Orbital elements (a, e, i, Omega, omega) for two confocal ellipses
         -> orthonormal in-plane basis vectors P, Q (and p, q for orbit 2)
         -> K, L, M, N = pairwise dot products between the two orbital bases.
      2. K, L, M, N, a1, e1, a2, e2 -> coefficients A1..A15 of the gradient
         system (stationarity conditions of d^2).
      3. Weierstrass substitution t = tan(u1/2), s = tan(u2/2) turns the
         trigonometric system into an ordinary bivariate polynomial system:
         p(t,s) = 0 (quadratic in s), q(t,s) = 0 (quartic in s).
      4. Sylvester resultant of p, q w.r.t. s -> det(S0(t)), degree 20 in t.
         Computed here by EXACT polynomial cofactor expansion (numpy.poly1d
         arithmetic), not FFT and not sympy (sympy's fraction-free Bareiss
         determinant blew up numerically on float coefficients here).
      5. det(S0(t)) factors as (1+t^2)^2 * (degree-16 polynomial): the
         (1+t^2)^2 part is 4 always-present, non-real roots (t = +-i, each
         double), an artifact of the tangent-half-angle substitution, not
         genuine stationary points. Divide it out (numpy.polydiv).
      6. Real roots of the degree-16 quotient are the t-components of all
         stationary points of d^2. For each, recover s via the (easy,
         quadratic) equation p(t,s)=0, picking whichever of its two roots
         also (numerically) satisfies q(t,s)=0.
      7. Convert (t, s) -> (u1, u2) -> Cartesian positions -> distances.
         The MOID is the smallest of these.
    """

    REAL_TOL = 1e-6

    def __init__(self, verbose=False):
        self.verbose = verbose

    def compute(self, elements1, elements2, verbose=False):
        "Returns MOIDResult."

        a1, e1 = elements1["a"], elements1["e"]
        a2, e2 = elements2["a"], elements2["e"]

        P, Q = self._orbital_basis(elements1["i"], elements1["raan"], elements1["argp"])
        p, q = self._orbital_basis(elements2["i"], elements2["raan"], elements2["argp"])

        K = float(np.dot(P, p))
        L = float(np.dot(Q, p))
        M = float(np.dot(P, q))
        N = float(np.dot(Q, q))

        Ac = self._build_A_coeffs(a1, e1, a2, e2, K, L, M, N)
        alpha, beta, gamma, Aexpr, Bexpr, Dexpr = self._build_polynomials(Ac)

        S0 = self._sylvester_S0(alpha, beta, gamma, Aexpr, Bexpr, Dexpr)
        det_S0 = self._poly_det(S0)

        if verbose:
            print("degree of raw resultant:", det_S0.order)

        known_factor = np.poly1d([1.0, 0.0, 1.0]) ** 2  # (1+t^2)^2
        quotient, remainder = np.polydiv(det_S0.coeffs, known_factor.coeffs)
        if verbose:
            print(
                "max |remainder| coeff after dividing by (1+t^2)^2:",
                np.max(np.abs(remainder)) if len(remainder) else 0.0,
            )
            print("degree of reduced polynomial:", len(quotient) - 1)

        roots_t = np.roots(quotient)

        def cart_pos(a, e, P_, Q_, u):
            x = a * (np.cos(u) - e)
            y = a * np.sqrt(1 - e**2) * np.sin(u)
            return x * P_ + y * Q_

        candidates = []
        for tv in roots_t:
            if abs(tv.imag) > self.REAL_TOL:
                continue
            tv_r = tv.real

            al, be, ga = alpha(tv_r), beta(tv_r), gamma(tv_r)
            Av, Bv, Dv = Aexpr(tv_r), Bexpr(tv_r), Dexpr(tv_r)

            if abs(al) < 1e-12:
                continue
            disc = be**2 - 4 * al * ga
            sroot = np.sqrt(complex(disc))
            s_candidates = [(-be + sroot) / (2 * al), (-be - sroot) / (2 * al)]

            best_s, best_res = None, None
            for sv in s_candidates:
                qval = Av * sv**4 + Bv * sv**3 + Dv * sv - Av
                if best_res is None or abs(qval) < best_res:
                    best_res = abs(qval)
                    best_s = sv

            if best_s is None or abs(best_s.imag) > self.REAL_TOL:
                continue

            s_r = best_s.real
            u1 = 2 * np.arctan(tv_r)
            u2 = 2 * np.arctan(s_r)

            X1 = cart_pos(a1, e1, P, Q, u1)
            X2 = cart_pos(a2, e2, p, q, u2)
            dist = np.linalg.norm(X1 - X2)
            candidates.append((dist, u1, u2, best_res))

        if not candidates:
            raise RuntimeError("No real stationary points found -- check inputs.")

        candidates.sort(key=lambda c: c[0])
        moid = candidates[0][0]
        return moid, candidates

    @staticmethod
    def _orbital_basis(i_deg, raan_deg, argp_deg):
        """Return the (P, Q) orthonormal in-plane basis vectors of one ellipse."""
        i = np.radians(i_deg)
        Om = np.radians(raan_deg)
        om = np.radians(argp_deg)

        Px = np.cos(om) * np.cos(Om) - np.cos(i) * np.sin(om) * np.sin(Om)
        Py = np.cos(om) * np.sin(Om) + np.cos(i) * np.sin(om) * np.cos(Om)
        Pz = np.sin(om) * np.sin(i)

        Qx = -np.sin(om) * np.cos(Om) - np.cos(i) * np.cos(om) * np.sin(Om)
        Qy = -np.sin(om) * np.sin(Om) + np.cos(i) * np.cos(om) * np.cos(Om)
        Qz = np.cos(om) * np.sin(i)

        return np.array([Px, Py, Pz]), np.array([Qx, Qy, Qz])

    @staticmethod
    def _build_A_coeffs(a1, e1, a2, e2, K, L, M, N):
        A1 = a1**2 * (1 - e1**2)
        A3 = a1**2
        A4 = a2**2 * (1 - e2**2)
        A6 = a2**2
        A7 = -2 * a1 * a2 * np.sqrt(1 - e1**2) * np.sqrt(1 - e2**2) * N
        A8 = -2 * a1 * a2 * np.sqrt(1 - e1**2) * L
        A9 = -2 * a1 * a2 * np.sqrt(1 - e2**2) * M
        A10 = -2 * a1 * a2 * K
        A11 = 2 * a1 * a2 * e2 * np.sqrt(1 - e1**2) * L
        A12 = 2 * a1 * (a2 * e2 * K - a1 * e1)
        A13 = 2 * a1 * a2 * e1 * np.sqrt(1 - e2**2) * M
        A14 = 2 * a2 * (a1 * e1 * K - a2 * e2)
        return dict(
            A1=A1,
            A3=A3,
            A4=A4,
            A6=A6,
            A7=A7,
            A8=A8,
            A9=A9,
            A10=A10,
            A11=A11,
            A12=A12,
            A13=A13,
            A14=A14,
        )

    @staticmethod
    def _build_polynomials(Ac):
        """Return alpha, beta, gamma, A, B, D as numpy.poly1d objects in t."""
        A1, A3, A4, A6 = Ac["A1"], Ac["A3"], Ac["A4"], Ac["A6"]
        A7, A8, A9, A10 = Ac["A7"], Ac["A8"], Ac["A9"], Ac["A10"]
        A11, A12, A13, A14 = Ac["A11"], Ac["A12"], Ac["A13"], Ac["A14"]

        # poly1d coefficients are given highest-degree first.
        alpha = np.poly1d(
            [
                (A8 - A11),
                (-4 * A1 + 4 * A3 + 2 * A10 - 2 * A12),
                0.0,
                (4 * A1 - 4 * A3 + 2 * A10 - 2 * A12),
                (A11 - A8),
            ]
        )
        beta = np.poly1d([-2 * A7, -4 * A9, 0.0, -4 * A9, 2 * A7])
        gamma = np.poly1d(
            [
                -(A8 + A11),
                (-4 * A1 + 4 * A3 - 2 * A10 - 2 * A12),
                0.0,
                (4 * A1 - 4 * A3 - 2 * A10 - 2 * A12),
                (A11 + A8),
            ]
        )
        Aexpr = np.poly1d([(A9 - A13), -2 * A7, -(A9 + A13)])
        Bexpr = np.poly1d(
            [
                (-4 * A4 + 4 * A6 + 2 * A10 - 2 * A14),
                -4 * A8,
                (-4 * A4 + 4 * A6 - 2 * A10 - 2 * A14),
            ]
        )
        Dexpr = np.poly1d(
            [(4 * A4 - 4 * A6 + 2 * A10 - 2 * A14), -4 * A8, (4 * A4 - 4 * A6 - 2 * A10 - 2 * A14)]
        )
        return alpha, beta, gamma, Aexpr, Bexpr, Dexpr

    @staticmethod
    def _poly_det(M):
        """Exact determinant of a matrix of numpy.poly1d entries via cofactor expansion."""
        n = len(M)
        if n == 1:
            return M[0][0]
        if n == 2:
            return M[0][0] * M[1][1] - M[0][1] * M[1][0]
        det = np.poly1d([0.0])
        for j in range(n):
            minor = [row[:j] + row[j + 1 :] for row in M[1:]]
            cofactor = M[0][j] * AlgebraicMOIDSolver._poly_det(minor)
            det = det + cofactor if j % 2 == 0 else det - cofactor
        return det

    @staticmethod
    def _sylvester_S0(alpha, beta, gamma, A, B, D):
        Z = np.poly1d([0.0])
        return [
            [alpha, Z, Z, Z, A, Z],
            [beta, alpha, Z, Z, B, A],
            [gamma, beta, alpha, Z, Z, B],
            [Z, gamma, beta, alpha, D, Z],
            [Z, Z, gamma, beta, -A, D],
            [Z, Z, Z, gamma, Z, -A],
        ]
