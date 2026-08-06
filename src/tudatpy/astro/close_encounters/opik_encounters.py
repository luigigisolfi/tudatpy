import numpy as np
import math
from tudatpy.dynamics import environment_setup


class OpikEncounter:

    GRAVITATIONAL_CONSTANT = 6.6743e-11  # [m^3 kg^-1 s^-2]

    def __init__(
        self,
        small_body_orbital_elements,  # tudat: [a(m), e, i(rad), argp(rad), raan(rad), true_anomaly(rad)]
        planet_orbital_elements,  # tudat: [a(m), e, i(rad), argp(rad), raan(rad), true_anomaly(rad)]
        central_body_mass,  # [KG] planet physical mass
        central_body_radius,  # [m] planet's physical radius
        small_body_epoch=0.0,
        planet_epoch=0.0,
        encounter_epoch=None,
        frame_origin="Sun",
        frame_orientation="ECLIPJ2000",
    ):
        self.small_body_orbital_elements = small_body_orbital_elements
        self.planet_orbital_elements = planet_orbital_elements
        self.small_body_epoch = small_body_epoch
        self.planet_epoch = planet_epoch
        self.encounter_epoch = encounter_epoch
        self.frame_origin = frame_origin
        self.frame_orientation = frame_orientation

        self.central_body_mass = central_body_mass  # [KG]
        self.central_body_radius = central_body_radius  # [m]
        self.solar_mass = 1.988475e30  # [KG]

    @property
    def sun_gravitational_parameter(self):
        """mu of the Sun [m^3/s^2], used as the central-body GM for both Kepler ephemerides."""
        return self.GRAVITATIONAL_CONSTANT * self.solar_mass

    @property
    def central_body_orbital_radius(self):
        """Planet's heliocentric semi-major axis [m], from planet_orbital_elements."""
        return self.planet_orbital_elements[0]

    @property
    def a(self):
        """Opik-normalized small-body semi-major axis: a_body / a_planet (dimensionless)."""
        return self.small_body_orbital_elements[0] / self.central_body_orbital_radius

    @property
    def e(self):
        return self.small_body_orbital_elements[1]

    @property
    def i(self):
        return self.small_body_orbital_elements[2]

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
        phi = np.arctan2(Ux, Uz)
        return [theta, phi]

    @property
    def impact_parameter(self):
        Q = self.central_body_radius / self.central_body_orbital_radius  # both in meters
        U = self.planetocentric_velocity_magnitude
        S = math.sqrt(2 * self.central_body_mass / (self.solar_mass * Q))  # mass -> ratio
        return Q * math.sqrt(1 + S**2 / U**2)

    @property
    def deflection_angle(self):
        U = self.planetocentric_velocity_magnitude
        b = self.impact_parameter
        if self.tisserand_parameter > 3:
            raise ValueError(
                f"Tisserand Parameter value is {self.tisserand_parameter}. "
                f"Opik Theory is not valid in this regime."
            )
        return 2 * np.arctan(self.central_body_mass / (b * U**2))

    def _build_kepler_ephemeris(self, tudat_elements, epoch, body_name):
        if tudat_elements is None or len(tudat_elements) < 6:
            raise ValueError(
                f"{body_name} orbital elements must have all 6 tudat-format components "
                "[a (m), e, i (rad), argument_of_periapsis (rad), raan (rad), "
                "true_anomaly (rad)] to build a tudatpy Kepler ephemeris."
            )
        settings = environment_setup.ephemeris.keplerian(
            tudat_elements,
            epoch,
            self.sun_gravitational_parameter,
            self.frame_origin,
            self.frame_orientation,
        )
        return environment_setup.create_body_ephemeris(settings, body_name)

    @property
    def body_ephemeris(self):
        """tudatpy KeplerEphemeris for the small body, built from
        small_body_orbital_elements at small_body_epoch."""
        return self._build_kepler_ephemeris(
            self.small_body_orbital_elements, self.small_body_epoch, "SmallBody"
        )

    @property
    def planet_ephemeris(self):
        """tudatpy KeplerEphemeris for the planet, built from
        planet_orbital_elements at planet_epoch."""
        return self._build_kepler_ephemeris(
            self.planet_orbital_elements, self.planet_epoch, "Planet"
        )

    @property
    def encounter_state_vectors(self):
        """r_body, v_body, r_planet, v_planet at self.encounter_epoch (the MOID
        time), evaluated from the two Kepler ephemerides above."""
        if self.encounter_epoch is None:
            raise ValueError(
                "encounter_epoch (time of MOID) must be set to evaluate the ephemerides."
            )

        body_state = np.asarray(self.body_ephemeris.cartesian_state(self.encounter_epoch)).flatten()
        planet_state = np.asarray(
            self.planet_ephemeris.cartesian_state(self.encounter_epoch)
        ).flatten()

        r_body, v_body = body_state[:3], body_state[3:6]
        r_planet, v_planet = planet_state[:3], planet_state[3:6]
        return r_body, v_body, r_planet, v_planet

    # ------------------------------------------------------------------
    # psi / outgoing state
    # ------------------------------------------------------------------

    def psi(self, theta, phi, r_body, v_body, r_planet, v_planet):
        """
        Computes psi in the tangent plane at the incoming
        asymptote direction (Tangent plane of U), from real position/velocity vectors.
        """

        r_body, v_body, r_planet, v_planet = map(np.array, (r_body, v_body, r_planet, v_planet))

        X = r_planet / np.linalg.norm(r_planet)  # X axis
        Z = np.cross(r_planet, v_planet)  # Z axis
        Z /= np.linalg.norm(Z)  # Z axis
        Y = np.cross(Z, X)  # Y axis

        to_frame = lambda v: np.array(
            [v @ X, v @ Y, v @ Z]
        )  # projectss vectors onto planet's reference frame

        r_rel = to_frame(r_body - r_planet)  # yields relative position
        v_rel = to_frame(v_body - v_planet)  # yields relative velocity [Ux, Uy, Uz]

        U_hat = v_rel / np.linalg.norm(v_rel)  # relative velocity unit vector
        h_hat = np.cross(r_rel, v_rel)  # angular momentum vector
        h_hat /= np.linalg.norm(h_hat)  # angular momentum unit vector
        b_hat = np.cross(h_hat, U_hat)  # impact parameter unit vector (this is tangent to U space)

        # decompose b in theta and phi components
        du_dtheta = np.array(
            [np.cos(theta) * np.sin(phi), -np.sin(theta), np.cos(theta) * np.cos(phi)]
        )
        du_dphi = np.array([np.cos(phi), 0, -np.sin(phi)])

        # compute arctan(dU/dphi)/(-dU/dtheta) = psi
        return np.arctan2(b_hat @ du_dphi, -(b_hat @ du_dtheta)), v_rel

    @property
    def outgoing_asymptote_angles_vector(self):
        """theta', phi' (plus psi, gamma, and the reproduced v_rel for sanity
        checking), computed from real state vectors pulled out of
        encounter_state_vectors (tudatpy Kepler ephemerides at encounter_epoch)."""
        r_body, v_body, r_planet, v_planet = self.encounter_state_vectors

        gamma = self.deflection_angle
        theta, phi = self.incoming_asymptote_angles_vector

        psi, v_rel_check = self.psi(theta, phi, r_body, v_body, r_planet, v_planet)

        cos_theta_out = np.cos(theta) * np.cos(gamma) + np.sin(theta) * np.sin(gamma) * np.cos(psi)
        theta_out = np.arccos(cos_theta_out)

        sin_chi_num = np.sin(gamma) * np.sin(psi)
        cos_chi_den = np.sin(theta) * np.cos(gamma) - np.cos(theta) * np.sin(gamma) * np.cos(psi)
        chi = np.arctan2(sin_chi_num, cos_chi_den)

        phi_out = phi - chi
        return [theta_out, phi_out], psi, gamma, v_rel_check

    @property
    def outgoing_orbital_elements(self):
        """
        Computes outgoing orbital elements computation as in Carusi et al, 1990
        Post-encounter (a', e', i'). Opik-normalized, same convention as self.a/e/i
        -- derived from the outgoing_asymptote_angles_vector property (theta', phi')
        plus |U| (unchanged through the encounter). Inverts planetocentric_velocity_vector:

            Ux = U sin(theta) sin(phi)
            Uy = U cos(theta)
            Uz = U sin(theta) cos(phi)

            a  = 1 / (1 - U^2 - 2*Uy)                      [matches eq. (13)'s denominator]
            e  = sqrt(1 - [(1+Uy)^2 + Uz^2] / a)
            i  = atan2(Uz, 1+Uy)
        """
        (theta_out, phi_out), psi, gamma, v_rel_check = self.outgoing_asymptote_angles_vector

        U = self.planetocentric_velocity_magnitude

        # Ux_out = U * np.sin(theta_out) * np.sin(phi_out) # NOT NEEDED
        Uy_out = U * np.cos(theta_out)
        Uz_out = U * np.sin(theta_out) * np.cos(phi_out)

        a_out = 1 / (1 - U**2 - 2 * Uy_out)
        e_out = math.sqrt(max(0.0, 1 - ((1 + Uy_out) ** 2 + Uz_out**2) / a_out))
        i_out = np.arctan2(Uz_out, 1 + Uy_out)

        return a_out, e_out, i_out


if __name__ == "__main__":
    AU = 1.495978707e11  # m

    central_body_mass = 5.9722e24  # Earth mass, KG (actual mass, not a ratio)
    central_body_radius = 6371.0e3  # Earth radius, meters

    # Tudat-format orbital elements: [a (m), e, i (rad), argp (rad), raan (rad), true_anomaly (rad)]
    small_body_orbital_elements = [1.2 * AU, 0.30, math.radians(8.0), 0.5, 1.0, 0.2]
    planet_orbital_elements = [1.0 * AU, 0.0167, 0.0, 1.8, 0.0, 0.3]

    enc = OpikEncounter(
        small_body_orbital_elements,
        planet_orbital_elements,
        central_body_mass,
        central_body_radius,
        small_body_epoch=0.0,
        planet_epoch=0.0,
        encounter_epoch=0.0,
    )

    print("--- Incoming ---")
    print(f"a, e, i (Opik-normalized) = {enc.a:.6f}, {enc.e:.6f}, {math.degrees(enc.i):.4f} deg")
    print(f"Tisserand T         = {enc.tisserand_parameter:.6f}")
    print(f"|U|                 = {enc.planetocentric_velocity_magnitude:.6f}")
    print(f"Ux, Uy, Uz          = {enc.planetocentric_velocity_vector}")
    theta, phi = enc.incoming_asymptote_angles_vector
    print(f"theta, phi (deg)    = {math.degrees(theta):.4f}, {math.degrees(phi):.4f}")
    print(f"impact parameter b  = {enc.impact_parameter:.6e} (normalized)")
    print(f"deflection gamma    = {math.degrees(enc.deflection_angle):.4f} deg")

    print("\n--- Outgoing (tudatpy-backed) ---")
    (theta_out, phi_out), psi, gamma, v_rel_check = enc.outgoing_asymptote_angles_vector
    print(f"psi (deg)           = {math.degrees(psi):.4f}")
    print(f"theta', phi' (deg)  = {math.degrees(theta_out):.4f}, {math.degrees(phi_out):.4f}")
    print(f"v_rel (should ~ Ux,Uy,Uz) = {v_rel_check}")

    a_out, e_out, i_out = enc.outgoing_orbital_elements
    print(f"a' (Opik-normalized) = {a_out:.6f}")
    print(f"e'                    = {e_out:.6f}")
    print(f"i' (deg)              = {math.degrees(i_out):.4f}")
