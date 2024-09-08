import numpy as np
from scipy.optimize import linear_sum_assignment, minimize
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
from collections import deque
import matplotlib.pyplot as plt
from ai_economist.foundation.base.base_agent import BaseAgent, agent_registry
from ai_economist.foundation.entities import resource_registry

@agent_registry.add
class CentralPlanner(BaseAgent):
    name = "CentralPlanner"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._idx = "p"  # Planner index
        
        self.planning_horizon = 100
        
        self.objective_weights = {
            "production": 0.3,
            "equality": 0.3,
            "sustainability": 0.2,
            "innovation": 0.2
        }
        
        self.current_plans = {
            "production_targets": {},
            "resource_allocation": {},
            "distribution_quotas": {}
        }
        
        self.historical_data = {
            "production": deque(maxlen=1000),
            "equality": deque(maxlen=1000),
            "sustainability": deque(maxlen=1000),
            "innovation": deque(maxlen=1000)
        }
        
        self.sustainability_thresholds = {}
        self.innovation_investments = {}

    def generate_plan(self, world_state):
        production_targets = self._set_production_targets(world_state)
        resource_allocation = self._allocate_resources(world_state)
        distribution_quotas = self._set_distribution_quotas(world_state)
        
        self.current_plans = {
            "production_targets": production_targets,
            "resource_allocation": resource_allocation,
            "distribution_quotas": distribution_quotas
        }

    def _set_production_targets(self, world_state):
        targets = {}
        for resource in resource_registry.entries:
            historical_data = [p.get(resource, 0) for p in self.historical_data["production"]]
            if len(historical_data) > 10:
                model = ARIMA(historical_data, order=(1, 1, 1))
                results = model.fit()
                forecast = results.forecast(steps=1)[0]
                targets[resource] = max(int(forecast), 1)
            else:
                current_amount = sum(agent.inventory[resource] for agent in world_state.agents)
                targets[resource] = int(current_amount * 1.1)  # 10% growth as default
        return targets

    def _allocate_resources(self, world_state):
        allocation = {agent.idx: {} for agent in world_state.agents}
        for resource in resource_registry.entries:
            total_resource = sum(agent.inventory[resource] for agent in world_state.agents)
            agent_skills = [agent.state.get("skill", 1) for agent in world_state.agents]
            
            def objective(x):
                return -np.sum(np.multiply(x, agent_skills))
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - total_resource},
                {'type': 'ineq', 'fun': lambda x: x}  # Non-negativity constraint
            ]
            
            initial_guess = np.ones(len(world_state.agents)) * (total_resource / len(world_state.agents))
            result = minimize(objective, initial_guess, method='SLSQP', constraints=constraints)
            
            for i, agent in enumerate(world_state.agents):
                allocation[agent.idx][resource] = result.x[i]
        
        return allocation

    def _set_distribution_quotas(self, world_state):
        quotas = {agent.idx: {} for agent in world_state.agents}
        for resource in resource_registry.entries:
            total_production = self.current_plans["production_targets"][resource]
            agent_needs = self._estimate_agent_needs(resource, world_state)
            
            total_needs = sum(agent_needs.values())
            for agent in world_state.agents:
                quotas[agent.idx][resource] = (total_production * agent_needs[agent.idx] / total_needs) if total_needs > 0 else 0
        
        return quotas

    def _estimate_agent_needs(self, resource, world_state):
        needs = {}
        for agent in world_state.agents:
            current_inventory = agent.inventory[resource]
            skill_level = agent.state.get("skill", 1)
            historical_usage = self._get_historical_usage(agent.idx, resource)
            
            estimated_need = (historical_usage * skill_level * 1.1) - current_inventory  # 10% buffer
            needs[agent.idx] = max(estimated_need, 0)  # Ensure non-negative need
        return needs

    def _get_historical_usage(self, agent_idx, resource):
        # Implement logic to retrieve and calculate historical usage
        return 10  # Placeholder

    def optimize_plan(self, world_state):
        current_score = self._evaluate_plan(world_state)
        
        for _ in range(100):  # Increased optimization attempts
            new_plan = self._generate_altered_plan()
            new_score = self._evaluate_plan(world_state, new_plan)
            
            if new_score > current_score:
                self.current_plans = new_plan
                current_score = new_score

    def _generate_altered_plan(self):
        new_plan = deepcopy(self.current_plans)
        
        # Randomly adjust production targets
        for resource in new_plan["production_targets"]:
            adjustment = np.random.uniform(0.9, 1.1)
            new_plan["production_targets"][resource] *= adjustment
        
        # Randomly redistribute some resources
        for resource in new_plan["resource_allocation"]:
            total = sum(new_plan["resource_allocation"][agent][resource] for agent in new_plan["resource_allocation"])
            redistribution = np.random.dirichlet([1]*len(new_plan["resource_allocation"]))
            for i, agent in enumerate(new_plan["resource_allocation"]):
                new_plan["resource_allocation"][agent][resource] = total * redistribution[i]
        
        return new_plan

    def _evaluate_plan(self, world_state, plan=None):
        if plan is None:
            plan = self.current_plans
        
        production_score = self._evaluate_production(world_state, plan)
        equality_score = self._evaluate_equality(world_state, plan)
        sustainability_score = self._evaluate_sustainability(world_state, plan)
        innovation_score = self._evaluate_innovation(world_state, plan)
        
        total_score = (
            self.objective_weights["production"] * production_score +
            self.objective_weights["equality"] * equality_score +
            self.objective_weights["sustainability"] * sustainability_score +
            self.objective_weights["innovation"] * innovation_score
        )
        
        return total_score

    def _evaluate_production(self, world_state, plan):
        target_production = sum(plan["production_targets"].values())
        current_production = sum(sum(agent.inventory[r] for r in resource_registry.entries) 
                                 for agent in world_state.agents)
        return min(current_production / target_production, 1)

    def _evaluate_equality(self, world_state, plan):
        agent_wealth = [sum(agent.inventory[r] for r in resource_registry.entries) 
                        for agent in world_state.agents]
        gini_coefficient = self._calculate_gini(agent_wealth)
        theil_index = self._calculate_theil(agent_wealth)
        return 1 - (0.5 * gini_coefficient + 0.5 * theil_index)

    def _calculate_gini(self, array):
        array = sorted(array)
        index = np.arange(1, len(array) + 1)
        return (np.sum((2 * index - len(array) - 1) * array)) / (len(array) * np.sum(array))

    def _calculate_theil(self, array):
        n = len(array)
        mean = np.mean(array)
        return np.sum((array / (n * mean)) * np.log(array / mean))

    def _evaluate_sustainability(self, world_state, plan):
        sustainability_score = 0
        for resource in resource_registry.entries:
            current_amount = sum(agent.inventory[resource] for agent in world_state.agents)
            threshold = self.sustainability_thresholds.get(resource, current_amount * 0.5)
            sustainability_score += min(current_amount / threshold, 1)
        return sustainability_score / len(resource_registry.entries)

    def _evaluate_innovation(self, world_state, plan):
        total_wealth = sum(sum(agent.inventory[r] for r in resource_registry.entries) 
                           for agent in world_state.agents)
        innovation_investment = sum(self.innovation_investments.values())
        return min(innovation_investment / (0.1 * total_wealth), 1)  # Assume 10% of total wealth is ideal investment

    def get_action(self, world_state):
        if world_state.timestep % self.planning_horizon == 0:
            self.generate_plan(world_state)
            self.optimize_plan(world_state)
        
        return self._generate_action_from_plan(world_state)

    def _generate_action_from_plan(self, world_state):
        actions = {}
        for agent in world_state.agents:
            actions[agent.idx] = {
                "produce": self.current_plans["production_targets"],
                "collect": self.current_plans["resource_allocation"][agent.idx],
                "distribute": self.current_plans["distribution_quotas"][agent.idx]
            }
        return actions

    def update_historical_data(self, world_state):
        self.historical_data["production"].append({r: sum(agent.inventory[r] for agent in world_state.agents) 
                                                   for r in resource_registry.entries})
        self.historical_data["equality"].append(self._evaluate_equality(world_state, self.current_plans))
        self.historical_data["sustainability"].append(self._evaluate_sustainability(world_state, self.current_plans))
        self.historical_data["innovation"].append(self._evaluate_innovation(world_state, self.current_plans))

    def visualize_plan(self):
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        
        # Production Targets
        axs[0, 0].bar(self.current_plans["production_targets"].keys(), self.current_plans["production_targets"].values())
        axs[0, 0].set_title('Production Targets')
        axs[0, 0].set_xlabel('Resources')
        axs[0, 0].set_ylabel('Target Amount')
        
        # Resource Allocation
        resources = list(self.current_plans["resource_allocation"][list(self.current_plans["resource_allocation"].keys())[0]].keys())
        agents = list(self.current_plans["resource_allocation"].keys())
        data = [[self.current_plans["resource_allocation"][agent][resource] for resource in resources] for agent in agents]
        axs[0, 1].imshow(data, cmap='viridis')
        axs[0, 1].set_title('Resource Allocation')
        axs[0, 1].set_xticks(range(len(resources)))
        axs[0, 1].set_yticks(range(len(agents)))
        axs[0, 1].set_xticklabels(resources)
        axs[0, 1].set_yticklabels(agents)
        
        # Historical Data
        for i, (key, data) in enumerate(self.historical_data.items()):
            if data:
                axs[1, i % 2].plot(data)
                axs[1, i % 2].set_title(f'Historical {key.capitalize()}')
                axs[1, i % 2].set_xlabel('Time')
                axs[1, i % 2].set_ylabel('Value')
        
        plt.tight_layout()
        plt.show()

    def generate_report(self):
        report = "Economic Performance Report\n"
        report += "===========================\n\n"
        
        report += "Production Targets:\n"
        for resource, target in self.current_plans["production_targets"].items():
            report += f"  {resource}: {target}\n"
        
        report += "\nResource Allocation:\n"
        for agent, allocation in self.current_plans["resource_allocation"].items():
            report += f"  Agent {agent}:\n"
            for resource, amount in allocation.items():
                report += f"    {resource}: {amount:.2f}\n"
        
        report += "\nPerformance Metrics:\n"
        for metric, data in self.historical_data.items():
            if data:
                report += f"  {metric.capitalize()}: {data[-1]:.2f}\n"
        
        return report
