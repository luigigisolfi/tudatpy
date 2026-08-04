import numpy as np
import math

class OpikEncounter:

    def __init__(self,
                 small_body_orbital_elements,
                 central_body_mass,
                 central_body_radius,
                 central_body_orbital_radius):

        self.a, self.e, self.i = small_body_orbital_elements[0:3]
        self.central_body_mass = central_body_mass
        self.central_body_radius = central_body_radius
        self.central_body_orbital_radius = central_body_orbital_radius
        self.solar_mass = 1.988475e30 # [KG]

    @property
    def tisserand_parameter(self):
        return 1/self.a + 2*math.sqrt(self.a*(1-self.e**2))*np.cos(self.i)
    @property
    def planetocentric_velocity_magnitude(self):
        return  math.sqrt(3 - self.tisserand_parameter)

    @property
    def planetocentric_velocity_vector(self):
        Ux = math.sqrt(2 - 1/self.a - self.a*(1-self.e**2))
        Uy = math.sqrt(self.a*(1-self.e**2))*np.cos(self.i) - 1
        Uz = math.sqrt(self.a*(1-self.e**2))*np.sin(self.i)
        return [Ux,Uy,Uz]

    @property
    def incoming_asymptote_angles_vector(self):
        Ux, Uy, Uz= self.planetocentric_velocity_vector
        U = self.planetocentric_velocity_magnitude
        theta =np.acos( Uy/U )
        phi = np.atan(Ux/Uz)

        return [theta, phi]

    @property
    def impact_parameter(self):
        # The following formulas are taken frokm Opik's original paper,
        # double check them
        Q = self.central_body_radius/self.central_body_orbital_radius
        S = math.sqrt(2*self.central_body_mass/(self.solar_mass*Q))
        U = self.planetocentric_velocity_magnitude
        return Q*math.sqrt(1 + S**2/U**2)

    #@property
    #def deflection_angle(self):



