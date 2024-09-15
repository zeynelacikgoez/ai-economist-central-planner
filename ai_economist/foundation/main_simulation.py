import matplotlib.pyplot as plt
from ai_economist.foundations.engine import EconomyEnvironment
from ai_economist.planner.central_planner import CentralPlanner
from ai_economist.agents.planned_economy_agent import PlannedEconomyAgent
from ai_economist.foundations.rewards import PlannedEconomyReward
from ai_economist.utils.visualization import plot_results
from ai_economist.data_loader import load_gdp_data
import logging
import numpy as np
import os

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_simulation(strategy: str, num_agents: int, num_episodes: int, num_timesteps: int):
    """
    Führt eine Simulation mit einer bestimmten Strategie durch.

    Args:
        strategy (str): Die gewählte Planungsstrategie.
        num_agents (int): Anzahl der Agenten.
        num_episodes (int): Anzahl der Episoden.
        num_timesteps (int): Anzahl der Zeitschritte pro Episode.

    Returns:
        Dict[str, List[float]]: Die gesammelten KPIs.
    """
    try:
        # Initialisierung der Umgebung
        environment = EconomyEnvironment()

        # Initialisierung des zentralen Planers mit der gewählten Strategie
        central_planner = CentralPlanner(environment, strategy=strategy)

        # Initialisierung der Agenten
        for i in range(num_agents):
            agent = PlannedEconomyAgent(agent_id=i, central_planner=central_planner)
            environment.agents.append(agent)

        # Initialisierung der Belohnungsfunktion
        reward_function = PlannedEconomyReward()

        # Daten für die Visualisierung
        equality_scores = []
        efficiency_scores = []
        total_rewards = []

        for episode in range(num_episodes):
            environment.reset()
            central_planner.reset()
            for timestep in range(num_timesteps):
                # Zentralen Plan generieren
                central_planner.generate_plan()

                # Wende den zentralen Plan an
                plan = central_planner.plan
                environment.apply_central_plan(plan)

                # Aktion der Agenten ausführen
                actions = [agent.action(environment.get_observation(agent)) for agent in environment.agents]
                environment.step(actions)

                # Belohnungen berechnen
                reward_function(environment, environment.agents)

                # Erfahrungen sammeln und Modell trainieren
                experiences = environment.collect_experiences()
                central_planner.train_model(experiences)

            # Sammle KPIs für die Visualisierung
            equality = reward_function.calculate_equality(environment.agents)
            efficiency = reward_function.calculate_efficiency(environment)
            total_reward = np.mean([agent.reward for agent in environment.agents])

            equality_scores.append(equality)
            efficiency_scores.append(efficiency)
            total_rewards.append(total_reward)

            logger.info(f"Episode {episode+1}/{num_episodes} - Strategie: {strategy} - Gleichheit: {equality:.4f}, Effizienz: {efficiency:.4f}, Durchschnittliche Belohnung: {total_reward:.4f}")

        return {
            'equality': equality_scores,
            'efficiency': efficiency_scores,
            'rewards': total_rewards
        }

    except Exception as e:
        logger.error(f"Fehler in der Simulation mit Strategie '{strategy}': {e}")
        return {}

def main():
    strategies = ['balance', 'equality', 'efficiency']
    num_agents = 10
    num_episodes = 100
    num_timesteps = 50

    results = {}

    for strategy in strategies:
        logger.info(f"Starte Simulation mit Strategie '{strategy}'...")
        strategy_results = run_simulation(strategy, num_agents, num_episodes, num_timesteps)
        results[strategy] = strategy_results

    # Visualisierung der Ergebnisse für jede Strategie
    for strategy, kpis in results.items():
        if kpis:
            plot_results(kpis['equality'], kpis['efficiency'], kpis['rewards'], strategy=strategy)

if __name__ == "__main__":
    main()
