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
    
        # This model uses 3 speed as velocity:
        # ----  v = v_max = 0.32 when omega is low and
        # ----  v = 0.85 * v_max = 0.272 when omega is high and 
        # ----  v = v_min = 0.1
        # ---- the idea behind it is to reduce a bit the linear v in the curve 
        # ---- once it got out of lane the idea is to reduce the linear v to try to localize itself 

        # v = 0.26 --- robot

        # v = 0.32 --- sub 10957


        if parameters.in_lane:
           v = 0.32
        else:
           v = 0.1

        # This model uses the error in the lane pose to predict a point ahead in the trajetory:
        # If the robot is considered in lane, just the angular error is considered

        s_phi = math.sin(-parameters.phi)
        c_phi = math.cos(-parameters.phi)

        if parameters.in_lane:
           d = 0
        else:
           d = parameters.d

        # a = 0.1 -- robot
        # a = 0.12 --- sub 10957

        a = 0.12

        w = 2 * v * (d * c_phi + a * s_phi)/(a**2+d**2)

        # if (abs(w) > 0.35):

        if (abs(w) > 0.7):
            v = v * 0.8


        return v, w
