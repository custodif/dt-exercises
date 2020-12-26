import numpy as np
import math


class PurePursuitLaneController:
    """
    The Lane Controller can be used to compute control commands from pose estimations.

    The control commands are in terms of linear and angular velocity (v, omega). The input are errors in the relative
    pose of the Duckiebot in the current lane.

    """

    def __init__(self, parameters):

        self.parameters = parameters

    def update_parameters(self, parameters):
        """Updates parameters of LaneController object.

            Args:
                parameters (:obj:`dict`): dictionary containing the new parameters for LaneController object.
        """
        self.parameters = parameters
        
    def get_velocity(self, parameters):
    
        s_phi = math.sin(-parameters.phi)
        c_phi = math.cos(-parameters.phi)
        
        d = -parameters.d

        v = 0.28
   
        x = 0.01
        y = (1/c_phi) * (d + x * s_phi)
        L = math.sqrt(x**2 + y**2)

        L_max = 0.18

        while L < L_max:
            x += 0.01
            y = (1/c_phi) * (d + x * s_phi)
            L = math.sqrt(x**2 + y**2)

        w = 2 * v * y/(L**2)

        return v, w
