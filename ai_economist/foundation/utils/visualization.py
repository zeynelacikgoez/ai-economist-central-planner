import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_results(equality: List[float], efficiency: List[float], rewards: List[float], strategy: str) -> None:
    """
    Plottet die Ergebnisse der Simulation für eine bestimmte Strategie.

    Args:
        equality (List[float]): Liste der Gleichheits-Scores.
        efficiency (List[float]): Liste der Effizienz-Scores.
        rewards (List[float]): Liste der durchschnittlichen Belohnungen.
        strategy (str): Die verwendete Strategie.
    """
    try:
        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        sns.lineplot(data=equality, label='Gleichheit', color='blue')
        plt.xlabel('Episode')
        plt.ylabel('Gleichheits-Score')
        plt.title(f'Gleichheitsentwicklung ({strategy})')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 2)
        sns.lineplot(data=efficiency, label='Effizienz', color='green')
        plt.xlabel('Episode')
        plt.ylabel('Effizienz-Score')
        plt.title(f'Effizienzentwicklung ({strategy})')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 3)
        sns.lineplot(data=rewards, label='Durchschnittliche Belohnung', color='orange')
        plt.xlabel('Episode')
        plt.ylabel('Belohnung')
        plt.title(f'Belohnungsentwicklung ({strategy})')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"Fehler bei der Visualisierung der Ergebnisse für Strategie '{strategy}': {e}")

def plot_income_distribution(agents: List['PlannedEconomyAgent']) -> None:
    """
    Plottet die Einkommensverteilung der Agenten.

    Args:
        agents (List['PlannedEconomyAgent']): Die Liste der Agenten.
    """
    try:
        incomes = [agent.total_income for agent in agents]
        sns.histplot(incomes, bins=20, kde=True, color='skyblue')
        plt.title('Einkommensverteilung')
        plt.xlabel('Einkommen')
        plt.ylabel('Anzahl der Agenten')
        plt.show()
    except Exception as e:
        logger.error(f"Fehler bei der Visualisierung der Einkommensverteilung: {e}")

def plot_production_distribution(agents: List['PlannedEconomyAgent']) -> None:
    """
    Plottet die Produktionsverteilung der Agenten.

    Args:
        agents (List['PlannedEconomyAgent']): Die Liste der Agenten.
    """
    try:
        production = [agent.production_quantity for agent in agents]
        sns.histplot(production, bins=20, kde=True, color='salmon')
        plt.title('Produktionsverteilung')
        plt.xlabel('Produktionsmenge')
        plt.ylabel('Anzahl der Agenten')
        plt.show()
    except Exception as e:
        logger.error(f"Fehler bei der Visualisierung der Produktionsverteilung: {e}")
