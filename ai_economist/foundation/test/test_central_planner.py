import unittest
from unittest.mock import MagicMock
from ai_economist.foundation.engine import EconomyEnvironment
from ai_economist.planner.central_planner import CentralPlanner
from ai_economist.agents.planned_economy_agent import PlannedEconomyAgent
import numpy as np

class TestCentralPlanner(unittest.TestCase):
    def setUp(self):
        self.environment = EconomyEnvironment()
        self.central_planner = CentralPlanner(self.environment)
        for i in range(5):
            agent = PlannedEconomyAgent(agent_id=i, central_planner=self.central_planner)
            self.environment.agents.append(agent)

    def test_generate_plan(self):
        # Mocking the model's predict method
        self.central_planner.model.predict = MagicMock(return_value=np.array([100, 50] * 5))
        self.central_planner.generate_plan()
        self.assertEqual(len(self.central_planner.plan), 5)
        for agent_id, action in self.central_planner.plan.items():
            self.assertIn('production_quantity', action)
            self.assertIn('resource_allocation', action)
            self.assertEqual(action['production_quantity'], 100)
            self.assertEqual(action['resource_allocation']['amount'], 50)

    def test_apply_central_plan(self):
        self.central_planner.model.predict = MagicMock(return_value=np.array([100, 50] * 5))
        self.central_planner.generate_plan()
        plan = self.central_planner.plan
        self.environment.apply_central_plan(plan)
        for agent in self.environment.agents:
            self.assertEqual(agent.production_quantity, 100)
            self.assertEqual(agent.resource_allocation['amount'], 50)

    def test_get_agent_action_invalid_id(self):
        action = self.central_planner.get_agent_action(999)  # Nicht existierender Agent
        self.assertEqual(action, self.central_planner.default_action())

    def test_train_model(self):
        self.central_planner.model.train = MagicMock()
        experiences = {
            'observations': [{
                'total_resources': 1000, 
                'agent_states': [
                    {'production_quantity': 100, 'resource_allocation': {'amount': 50}} 
                    for _ in range(5)
                ]
            }],
            'actions': [
                {'production_quantity': 100, 'resource_allocation': {'amount': 50}} 
                for _ in range(5)
            ],
            'rewards': [150 for _ in range(5)],
            'next_observations': [{
                'total_resources': 950, 
                'agent_states': [
                    {'production_quantity': 100, 'resource_allocation': {'amount': 50}} 
                    for _ in range(5)
                ]
            }]
        }
        self.central_planner.train_model(experiences)
        self.central_planner.model.train.assert_called_once()

if __name__ == '__main__':
    unittest.main()