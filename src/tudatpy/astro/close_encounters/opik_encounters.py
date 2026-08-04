import numpy as np
import math


class OpikEncounter:

    def __init__(
        self,
        small_body_orbital_elements,
        central_body_mass,
        central_body_radius,
        central_body_orbital_radius,
    ):

        self.a, self.e, self.i = small_body_orbital_elements[0:3]
        self.central_body_mass = central_body_mass
        self.central_body_radius = central_body_radius
        self.central_body_orbital_radius = central_body_orbital_radius
        self.solar_mass = 1.988475e30  # [KG] TODO: consider using SPICE for this constant

    @property
    def tisserand_parameter(self):
        return 1 / self.a + 2 * math.sqrt(self.a * (1 - self.e**2)) * np.cos(self.i)

    @property
    def planetocentric_velocity_magnitude(self):
        return math.sqrt(3 - self.tisserand_parameter)

    @property
    def planetocentric_velocity_vector(self):
        Ux = math.sqrt(2 - 1 / self.a - self.a * (1 - self.e**2))
        Uy = math.sqrt(self.a * (1 - self.e**2)) * np.cos(self.i) - 1
        Uz = math.sqrt(self.a * (1 - self.e**2)) * np.sin(self.i)
        return [Ux, Uy, Uz]

    @property
    def incoming_asymptote_angles_vector(self):
        Ux, Uy, Uz = self.planetocentric_velocity_vector
        U = self.planetocentric_velocity_magnitude
        theta = np.arccos(Uy / U)
        phi = np.arctan(Ux / Uz)

        return [theta, phi]

    @property
    def impact_parameter(self):
        # The following formulas are taken frokm Opik's original paper,
        # double check them
        Q = self.central_body_radius / self.central_body_orbital_radius
        S = math.sqrt(2 * self.central_body_mass / (self.solar_mass * Q))
        U = self.planetocentric_velocity_magnitude
        return Q * math.sqrt(1 + S**2 / U**2)

    @property
    def deflection_angle(self):
        # Formula from G. Tommei's PhD Thesis (eq. 2.2)
        U = self.planetocentric_velocity_magnitude
        b = self.impact_parameter

        if self.tisserand_parameter > 3:  # From G. Tommei
            raise ValueError(
                f"Tisserand Parameter value is {self.tisserand_parameter}. "
                f"Opik Theory is not valid in this regime."
            )
        else:
            return 2 * np.arctan(self.central_body_mass / (b * U**2))

    def psi(self, theta, phi, r_body, v_body, r_planet, v_planet):
        r_body, v_body, r_planet, v_planet = map(np.array, (r_body, v_body, r_planet, v_planet))

        # Öpik frame axes at the planet: X radial, Z orbit-normal, Y along-track
        X = r_planet / np.linalg.norm(r_planet)
        Z = np.cross(r_planet, v_planet)
        Z /= np.linalg.norm(Z)
        Y = np.cross(Z, X)

        to_frame = lambda v: np.array([v @ X, v @ Y, v @ Z])

        r_rel = to_frame(r_body - r_planet)
        v_rel = to_frame(v_body - v_planet)  # should reproduce [Ux, Uy, Uz]

        U_hat = v_rel / np.linalg.norm(v_rel)
        h_hat = np.cross(r_rel, v_rel)
        h_hat /= np.linalg.norm(h_hat)
        b_hat = np.cross(h_hat, U_hat)  # unit impact-parameter direction

        m_hat = np.array([np.cos(theta) * np.sin(phi), -np.sin(theta), np.cos(theta) * np.cos(phi)])
        e_hat = np.array([np.cos(phi), 0, -np.sin(phi)])

        return np.arctan2(b_hat @ e_hat, -(b_hat @ m_hat))

    @property
    def outgoing_asymptote_angles_vector(self):
        gamma = self.deflection_angle
        theta, phi = self.incoming_asymptote_angles_vector

        # To get psi from r_body, v_body, r_planet, v_planet, we should:
        # 1) compute the MOID
        # 2) compute the time of the MOID
        # 3) extract state from keplerian ephemeris of both bodies
        # 4) inject into psi computation
        psi = self.psi(theta, phi, r_body, v_body, r_planet, v_planet)

        cos_theta_out = np.cos(theta) * np.cos(gamma) + np.sin(theta) * np.sin(gamma) * np.cos(psi)
        theta_out = np.arccos(cos_theta_out)

        sin_chi_num = np.sin(gamma) * np.sin(psi)
        cos_chi_den = np.sin(theta) * np.cos(gamma) - np.cos(theta) * np.sin(gamma) * np.cos(psi)
        chi = np.arctan2(sin_chi_num, cos_chi_den)

        phi_out = phi - chi
        return [theta_out, phi_out]
