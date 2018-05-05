from tasks.task import Task


class EuclideanTask(Task):
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, init_angle_velocities=None, runtime=5., target_pos=None):
        Task.__init__(self, init_pose=init_pose, init_velocities=init_velocities,
                      init_angle_velocities=init_angle_velocities, runtime=runtime, target_pos=target_pos)

        position = self.sim.pose[0:3]
        self.init_distance =  self._get_distance(position)

    def _get_distance(self, position):
        return sum([abs(position[i] - self.target_pos[i]) ** 2 for i in range(0, 3)])


    def get_reward(self):
        """
        Euclidean based reward. Closer to the target but included instability on rotors as penalized behavior.
        :return:
        """
        position = self.sim.pose[0:3]
        distance = self._get_distance(position) * 1.0
        return 1 - (self.init_distance - distance) / self.init_distance