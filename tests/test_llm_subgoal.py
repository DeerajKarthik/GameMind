import unittest
from llm.subgoal_generator import SubGoalGenerator

class DummyLlamaClient:
    def __init__(self, config):
        pass
    def generate(self, prompt):
        return ['collect wood', 'build shelter', 'explore']

class TestableSubGoalGenerator(SubGoalGenerator):
    def __init__(self, config, dummy_llm):
        self.config = config
        self.llm = dummy_llm
        self.prompt_template = 'State or Goal:\n{input}\nSub-goals:\n1.\n2.\n3.'

class TestSubGoalGenerator(unittest.TestCase):
    def setUp(self):
        self.config = {'llm': {'prompt_template': 'default.txt'}}
        self.gen = TestableSubGoalGenerator(self.config, DummyLlamaClient(self.config))
    def test_generate_subgoals(self):
        subgoals = self.gen.generate_subgoals('survive 100 steps')
        self.assertEqual(len(subgoals), 3)
        self.assertIn('collect wood', subgoals)

if __name__ == '__main__':
    unittest.main() 