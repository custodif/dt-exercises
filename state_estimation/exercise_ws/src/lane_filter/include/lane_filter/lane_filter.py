
from collections import OrderedDict
from scipy.stats import multivariate_normal
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from math import floor, sqrt


class LaneFilterHistogramKF():
    """ Generates an estimate of the lane pose.

    TODO: Fill in the details

    Args:
        configuration (:obj:`List`): A list of the parameters for the filter

    """

    def __init__(self, **kwargs):
        param_names = [
            # TODO all the parameters in the default.yaml should be listed here.
            'mean_d_0',
            'mean_phi_0',
            'sigma_d_0',
            'sigma_phi_0',
            'delta_d',
            'delta_phi',
            'd_max',
            'd_min',
            'phi_max',
            'phi_min',
            'cov_v',
            'linewidth_white',
            'linewidth_yellow',
            'lanewidth',
            'min_max',
            'sigma_d_mask',
            'sigma_phi_mask',
            'range_min',
            'range_est',
            'range_max',
            'weight_max',
            'weight_min',
        ]

        for p_name in param_names:
            assert p_name in kwargs
            setattr(self, p_name, kwargs[p_name])


        self.encoder_resolution = 135
        self.wheel_radius = 0.0325
        self.baseline = 0.0
        self.initialized = True
        self.reset()

    def reset(self):
        self.mean_0 = [self.mean_d_0, self.mean_phi_0]
        self.cov_0 = [[self.sigma_d_0, 0], [0, self.sigma_phi_0]]
        self.belief = {'mean': self.mean_0, 'covariance': self.cov_0}

    def predict(self, dt, left_encoder_delta, right_encoder_delta):
        #TODO update self.belief based on right and left encoder data + kinematics

        if not self.initialized:
            return
        
        self.encoder_resolution = 135
        self.wheel_radius = 0.0325
        wheel_dist = 0.1
        wheel_radius = self.wheel_radius

        l = wheel_dist
        
        Vl = left_encoder_delta * 2. * np.pi * wheel_radius / (self.encoder_resolution * dt)
        Vr = right_encoder_delta * 2. * np.pi * wheel_radius / (self.encoder_resolution * dt)


        if Vl == Vr:
            v = Vl = Vr
            theta_displacement = 0
            d_dtheta = dt * v * np.cos(self.belief['mean'][1])

        else:
            w = (Vr - Vl) / l
            v = (Vr + Vl) / 2. 
            d = v / w 
            theta_displacement = w * dt 

            c1 = d * np.cos(theta_displacement)
            c2 = d * np.sin(theta_displacement)

            d_dtheta = np.sin(self.belief['mean'][1])*(c1 - d) + np.cos(self.belief['mean'][1])*c2      
            
        F = np.array([[1, -d_dtheta], [0, 1]])

        B = np.array([[0], [1]])
        u = np.array([theta_displacement])

        Q = np.array([[0.2, 0], [0, 0.2]])

        self.belief['mean'] = (F @ self.belief['mean'] + B @ u).tolist()        
        self.belief['covariance'] = (F @ self.belief['covariance'] @ F.T + Q).tolist()

    def update(self, segments):
        # prepare the segments for each belief array
        segmentsArray = self.prepareSegments(segments)
        # generate all belief arrays

        measurement_likelihood = self.generate_measurement_likelihood(segmentsArray)

        if measurement_likelihood is not None:

        # TODO: Parameterize the measurement likelihood as a Gaussian

            bin_d = np.arange(self.d_min+self.delta_d/2, self.d_max+self.delta_d/2, self.delta_d)
            mea_likelihood_d = np.sum(measurement_likelihood, axis=1)

            pos_d_ini = 0
            pos_d_fin = len(bin_d)   

            values = np.array(bin_d[pos_d_ini:pos_d_fin])
            weights = np.array(mea_likelihood_d[pos_d_ini:pos_d_fin]/np.sum(mea_likelihood_d[pos_d_ini:pos_d_fin]))

            mu_d = 0
            std_d = 0

            mu_d = np.average(values, weights=weights)
            std_d = np.average((values-mu_d)**2, weights=weights)

            if std_d < 0.001:
                std_d =  0.001

            bin_phi = np.arange(self.phi_min+self.delta_phi/2, self.phi_max+self.delta_phi/2, self.delta_phi)
            mea_likelihood_phi = np.sum(measurement_likelihood, axis=0)

            pos_phi_ini = 0
            pos_phi_fin = len(bin_phi)   

            values = np.array(bin_phi[pos_phi_ini:pos_phi_fin])
            weights = np.array(mea_likelihood_phi[pos_phi_ini:pos_phi_fin]/np.sum(mea_likelihood_phi[pos_phi_ini:pos_phi_fin]))

            mu_phi = 0
            std_phi = 0

            mu_phi = np.average(values, weights=weights)
            std_phi = np.average((values-mu_phi)**2, weights=weights)

            if std_phi < 0.001:
                std_phi =  0.001
        else:
            mu_d = self.belief['mean'][0]
            mu_phi = self.belief['mean'][1]
            std_d =  0.2
            std_phi =  0.2


        # TODO: Apply the update equations for the Kalman Filter to self.belief

        H = np.array([[1, 0], [0, 1]])
        R = np.array([[std_d, 0], [0, std_phi]])

        predicted_mu = np.array(self.belief['mean'])

        predicted_Sigma = np.array(self.belief['covariance'])

        z = np.array([mu_d + np.random.normal(loc=0.0, scale=std_d),
                      mu_phi + np.random.normal(loc=0.0, scale=std_phi)])

        residual_mean = z - H @ predicted_mu

        residual_covariance = H @ predicted_Sigma @ H.T + R 

        kalman_gain = predicted_Sigma @ H.T @ np.linalg.inv(residual_covariance)

        self.belief['mean'] = (predicted_mu + kalman_gain @ residual_mean).tolist()
        self.belief['covariance'] = (predicted_Sigma - kalman_gain @ H @ predicted_Sigma).tolist()

    def getEstimate(self):
        return self.belief

    def generate_measurement_likelihood(self, segments):

        if len(segments) == 0:
            return None

        grid = np.mgrid[self.d_min:self.d_max:self.delta_d,
                                    self.phi_min:self.phi_max:self.delta_phi]

        # initialize measurement likelihood to all zeros
        measurement_likelihood = np.zeros(grid[0].shape)

        for segment in segments:
            d_i, phi_i, l_i, weight = self.generateVote(segment)

            # if the vote lands outside of the histogram discard it
            if d_i > self.d_max or d_i < self.d_min or phi_i < self.phi_min or phi_i > self.phi_max:
                continue

            i = int(floor((d_i - self.d_min) / self.delta_d))
            j = int(floor((phi_i - self.phi_min) / self.delta_phi))
            measurement_likelihood[i, j] = measurement_likelihood[i, j] + weight

        if np.linalg.norm(measurement_likelihood) == 0:
            return None

        # lastly normalize so that we have a valid probability density function

        measurement_likelihood = measurement_likelihood / \
            np.sum(measurement_likelihood)

        return measurement_likelihood





    # generate a vote for one segment
    def generateVote(self, segment):
        p1 = np.array([segment.points[0].x, segment.points[0].y])
        p2 = np.array([segment.points[1].x, segment.points[1].y])
        t_hat = (p2 - p1) / np.linalg.norm(p2 - p1)

        n_hat = np.array([-t_hat[1], t_hat[0]])
        d1 = np.inner(n_hat, p1)
        d2 = np.inner(n_hat, p2)
        l1 = np.inner(t_hat, p1)
        l2 = np.inner(t_hat, p2)
        if (l1 < 0):
            l1 = -l1
        if (l2 < 0):
            l2 = -l2

        l_i = (l1 + l2) / 2
        d_i = (d1 + d2) / 2
        phi_i = np.arcsin(t_hat[1])

        weight = self.weight_min

        if segment.color == segment.WHITE:  # right lane is white
            if(p1[0] > p2[0]):  # right edge of white lane
                d_i = d_i - self.linewidth_white
            else:  # left edge of white lane

                d_i = - d_i

                phi_i = -phi_i
            d_i = d_i - self.lanewidth / 2

        elif segment.color == segment.YELLOW:  # left lane is yellow
            if (p2[0] > p1[0]):  # left edge of yellow lane
                d_i = d_i - self.linewidth_yellow
                phi_i = -phi_i
            else:  # right edge of white lane
                d_i = -d_i
            d_i = self.lanewidth / 2 - d_i

        # weight = distance
        weight = 1
        return d_i, phi_i, l_i, weight

    def get_inlier_segments(self, segments, d_max, phi_max):
        inlier_segments = []
        for segment in segments:
            d_s, phi_s, l, w = self.generateVote(segment)
            if abs(d_s - d_max) < 3*self.delta_d and abs(phi_s - phi_max) < 3*self.delta_phi:
                inlier_segments.append(segment)
        return inlier_segments

    # get the distance from the center of the Duckiebot to the center point of a segment
    def getSegmentDistance(self, segment):
        x_c = (segment.points[0].x + segment.points[1].x) / 2
        y_c = (segment.points[0].y + segment.points[1].y) / 2
        return sqrt(x_c**2 + y_c**2)

    # prepare the segments for the creation of the belief arrays
    def prepareSegments(self, segments):
        segmentsArray = []
        self.filtered_segments = []
        for segment in segments:

            # we don't care about RED ones for now
            if segment.color != segment.WHITE and segment.color != segment.YELLOW:
                continue
            # filter out any segments that are behind us
            if segment.points[0].x < 0 or segment.points[1].x < 0:
                continue

            self.filtered_segments.append(segment)
            # only consider points in a certain range from the Duckiebot for the position estimation
            point_range = self.getSegmentDistance(segment)
            if point_range < self.range_est:
                segmentsArray.append(segment)

        return segmentsArray