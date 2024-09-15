from ai_economist.foundations.rewards import RewardFunction
from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass

@dataclass
class PlannedEconomyReward(RewardFunction):
    """
    Belohnungsfunktion für die geplante Wirtschaft, die Gleichheit und Effizienz fördert.
    """

    def __call__(self, environment: 'EconomyEnvironment', agents: List['PlannedEconomyAgent']) -> None:
        """
        Berechnet und weist die Belohnung basierend auf Gleichheit und Effizienz zu.

        Args:
            environment (EconomyEnvironment): Die Wirtschaftsumgebung.
            agents (List[PlannedEconomyAgent]): Die Liste der Agenten.
        """
        try:
            equality_score = self.calculate_equality(agents)
            efficiency_score = self.calculate_efficiency(environment)
            total_reward = equality_score + efficiency_score
            for agent in agents:
                agent.reward = total_reward
        except Exception as e:
            print(f"Fehler bei der Berechnung der Belohnung: {e}")

    def calculate_equality(self, agents: List['PlannedEconomyAgent']) -> float:
        """
        Berechnet den Gleichheits-Score anhand des Gini-Koeffizienten.

        Args:
            agents (List['PlannedEconomyAgent']): Die Liste der Agenten.

        Returns:
            float: Der Gleichheits-Score.
        """
        incomes = np.array([agent.total_income for agent in agents])
        if len(incomes) == 0:
            return 0.0
        gini = self.gini_coefficient(incomes)
        return -gini  # Negativ, da geringere Ungleichheit besser ist

    def calculate_efficiency(self, environment: 'EconomyEnvironment') -> float:
        """
        Berechnet den Effizienz-Score basierend auf der Gesamtausstoß.

        Args:
            environment (EconomyEnvironment): Die Wirtschaftsumgebung.

        Returns:
            float: Der Effizienz-Score.
        """
        total_output = np.sum([agent.production_quantity for agent in environment.agents])
        return total_output

    def gini_coefficient(self, incomes: np.ndarray) -> float:
        """
        Berechnet den Gini-Koeffizienten.

        Args:
            incomes (np.ndarray): Die Einkommensdaten.

        Returns:
            float: Der Gini-Koeffizient.
        """
        if np.amin(incomes) < 0:
            raise ValueError("Incomes cannot be negative for Gini calculation.")
        incomes = incomes.flatten()
        if incomes.size == 0:
            return 0.0
        sorted_incomes = np.sort(incomes)
        n = incomes.size
        cumulative_income = np.cumsum(sorted_incomes)
        cumulative_income_sum = cumulative_income.sum()
        if cumulative_income_sum == 0:
            return 0.0
        gini = (2.0 * cumulative_income.sum() / cumulative_income_sum) - (n + 1) / n
        return gini
