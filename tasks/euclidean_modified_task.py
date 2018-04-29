from tasks.task import Task


class EuclideanModifiedTask(Task):
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, init_angle_velocities=None, runtime=5., target_pos=None):
        Task.__init__(self, init_pose=init_pose, init_velocities=init_velocities,
                      init_angle_velocities=init_angle_velocities, runtime=runtime, target_pos=target_pos)

    def get_reward(self):
        """
        Euclidean based reward. Closer to the target but included instability on rotors as penalized behavior.
        :return:
        """
        return 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()