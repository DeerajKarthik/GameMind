import unittest
import numpy as np
from planning.mcts import MCTS, MCTSNode

class DummyEnv:
    def __init__(self):
        self.action_space = type('A', (), {'n': 3})()
        self.env = self
        self.state = 0
    def step(self, action):
        return self.state + 1, 1.0, False, {}

class TestMCTS(unittest.TestCase):
    def setUp(self):
        self.env = DummyEnv()
        self.config = {'planning': {'mcts_simulations': 10, 'max_depth': 3}}
        self.mcts = MCTS(self.env, self.config)
    def test_search_returns_child(self):
        root_state = 0
        best = self.mcts.search(root_state)
        self.assertIsNotNone(best)
    def test_value_increases(self):
        root_state = 0
        self.mcts.simulations = 20
        best1 = self.mcts.search(root_state)
        v1 = best1.value
        self.mcts.simulations = 40
        best2 = self.mcts.search(root_state)
        v2 = best2.value
        self.assertTrue(v2 >= v1)

if __name__ == '__main__':
    unittest.main() 