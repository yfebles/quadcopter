from tasks import EuclideanTask


class EuclideanModifiedTask(EuclideanTask):
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, init_angle_velocities=None, runtime=5., target_pos=None):
        EuclideanTask.__init__(self, init_pose=init_pose, init_velocities=init_velocities,
                               init_angle_velocities=init_angle_velocities, runtime=runtime, target_pos=target_pos)

    def get_reward(self):
        """
        Euclidean based reward. Closer to the target but included instability on rotors as penalized behavior.
        :return:
        """
        position = self.sim.pose[0:3]
        distance = self._get_distance(position) * 1.0

        angles =self.sim.pose[3:]
        angles_distance = (angles.max() - angles.min()) / angles.max()

        return 1 - angles_distance - (self.init_distance - distance) / self.init_distance
