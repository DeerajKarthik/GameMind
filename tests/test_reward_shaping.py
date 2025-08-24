import unittest
from envs.crafter_wrapper import CrafterEnvWrapper

class TestRewardShaping(unittest.TestCase):
    def setUp(self):
        self.config = {'env': {'reward_shaping': True}}
        self.env = CrafterEnvWrapper(self.config)

    def test_achievement_bonus(self):
        self.env.prev_achievements = {ach: 0 for ach in self.env.achievements}
        info = {ach: 1 for ach in self.env.achievements}
        shaped = self.env._shape_reward(None, 0, info)
        self.assertTrue(shaped >= len(self.env.achievements))

    def test_exploration_bonus(self):
        self.env.prev_explored = 5
        info = {'explore': 10}
        shaped = self.env._shape_reward(None, 0, info)
        self.assertTrue(shaped > 0)

if __name__ == '__main__':
    unittest.main() 