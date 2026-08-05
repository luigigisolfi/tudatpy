import numpy as np
from tudatpy.astro.moid import AlgebraicMOIDSolver


def angular_difference(angle1, angle2):
    """Return smallest difference between two angles in degrees."""
    return abs((angle1 - angle2 + 180.0) % 360.0 - 180.0)


def test_gronchi_example():
    # Example from:
    # Grassi, Gronchi & Bau (2023),
    # "Revisiting the computation of the critical points of the Keplerian distance"

    q_gronchi_1 = 0.16582
    e_gronchi_1 = 0.84577

    q_gronchi_2 = 1.0
    e_gronchi_2 = 0.2

    a_1 = q_gronchi_1 / (1 - e_gronchi_1)
    a_2 = q_gronchi_2 / (1 - e_gronchi_2)

    paper_test = dict(
        a=a_1,
        e=e_gronchi_1,
        i=0.0,
        raan=0.0,
        argp=9.09466,
    )

    earth_test = dict(
        a=a_2,
        e=e_gronchi_2,
        i=0.0,
        raan=0.0,
        argp=10.0,
    )

    solver = AlgebraicMOIDSolver()
    moid, cands = solver.compute(paper_test, earth_test)

    # Reference stationary points from:
    # Grassi, Gronchi & Bau (2023)
    # (distance [AU], u1 [deg], u2 [deg])
    expected = [
        (0.0000000, 116.0625325, 153.9899286),
        (0.0000000, 243.6382848, 203.6865581),
        (0.4845432, 179.8948964, 178.9198966),
        (0.8341185, 1.6247542, 2.0946456),
        (0.8401907, 24.0090191, 38.3799855),
        (0.8445898, 334.2162041, 317.5202237),
        (1.6264123, 324.5270438, 126.4762243),
        (1.6334795, 34.8254033, 231.0377067),
        (1.6658557, 0.9077692, 180.7796090),
        (2.9845260, 179.9346562, 358.9929507),
    ]

    DIST_TOL = 5e-5  # AU
    ANGLE_TOL = 5e-3  # degrees

    recovered = 0

    for d_expected, u1_expected, u2_expected in expected:

        found = False

        for distance, u1, u2, _ in cands:
            u1_deg = np.degrees(u1) % 360.0
            u2_deg = np.degrees(u2) % 360.0

            if (
                abs(distance - d_expected) <= DIST_TOL
                and angular_difference(u1_deg, u1_expected) <= ANGLE_TOL
                and angular_difference(u2_deg, u2_expected) <= ANGLE_TOL
            ):
                found = True
                break

        if found:
            recovered += 1

    recovery_rate = recovered / len(expected)

    print(f"Recovered {recovered}/{len(expected)} " f"stationary points ({100*recovery_rate:.1f}%)")

    # Require at least 90% agreement with the reference table
    # (some solutions might differ due to different approach in root finding method, selecting different branches)
    assert recovery_rate >= 0.9

    # Check that the reported MOID corresponds to the smallest distance found
    np.testing.assert_allclose(
        moid,
        min(candidate[0] for candidate in cands),
        atol=1e-10,
    )
