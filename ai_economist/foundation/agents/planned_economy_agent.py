from ai_economist.agents.agent import Agent
from typing import Dict, Any

class PlannedEconomyAgent(Agent):
    """
    Agent in der geplanten Wirtschaft, der zentralen Planungsanweisungen folgt.

    Attributes:
        central_planner (CentralPlanner): Der zentrale Planer, der die Aktionen bestimmt.
        production_quantity (int): Die Produktionsmenge des Agenten.
        resource_allocation (Dict[str, Any]): Die Ressourcenallokation des Agenten.
        total_income (float): Das gesamte Einkommen des Agenten.
        reward (float): Die aktuelle Belohnung des Agenten.
    """

    def __init__(self, agent_id: int, central_planner: 'CentralPlanner', **kwargs):
        """
        Initialisiert einen neuen PlannedEconomyAgent.

        Args:
            agent_id (int): Die eindeutige ID des Agenten.
            central_planner (CentralPlanner): Der zentrale Planer.
            **kwargs: Zusätzliche Schlüsselwortargumente.
        """
        super().__init__(agent_id, **kwargs)
        self.central_planner = central_planner
        self.production_quantity: int = 0
        self.resource_allocation: Dict[str, Any] = {}
        self.total_income: float = 0.0
        self.reward: float = 0.0

    def action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Bestimmt die Aktion basierend auf der Beobachtung.

        Args:
            observation (Dict[str, Any]): Die aktuelle Beobachtung des Agenten.

        Returns:
            Dict[str, Any]: Die geplante Aktion.
        """
        try:
            planned_action = self.get_central_plan(observation)
            return planned_action
        except Exception as e:
            print(f"Fehler beim Abrufen des zentralen Plans für Agent {self.id}: {e}")
            return self.default_action()

    def get_central_plan(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ruft den zentralen Plan vom zentralen Planer ab.

        Args:
            observation (Dict[str, Any]): Die aktuelle Beobachtung des Agenten.

        Returns:
            Dict[str, Any]: Die geplante Aktion.
        """
        return self.central_planner.get_agent_action(self.id)

    def apply_action(self, action: Dict[str, Any]) -> None:
        """
        Wendet die vom zentralen Planer erhaltene Aktion an.

        Args:
            action (Dict[str, Any]): Die geplante Aktion.
        """
        try:
            self.production_quantity = action.get('production_quantity', 0)
            self.resource_allocation = action.get('resource_allocation', {})
            # Beispielhafte Einkommensberechnung
            self.total_income = self.production_quantity * 10 + self.resource_allocation.get('amount', 0) * 5
        except Exception as e:
            print(f"Fehler beim Anwenden der Aktion für Agent {self.id}: {e}")

    def default_action(self) -> Dict[str, Any]:
        """
        Definiert eine Standardaktion, falls kein spezifischer Plan vorhanden ist.

        Returns:
            Dict[str, Any]: Die Standardaktion.
        """
        return {
            'production_quantity': 0,
            'resource_allocation': {'resource_type': 'Rohstoff', 'amount': 0}
        }

    def get_state(self) -> Dict[str, Any]:
        """
        Gibt den aktuellen Zustand des Agenten zurück.

        Returns:
            Dict[str, Any]: Der aktuelle Zustand.
        """
        return {
            'production_quantity': self.production_quantity,
            'resource_allocation': self.resource_allocation
        }

    def get_observation(self) -> Dict[str, Any]:
        """
        Gibt die Beobachtung des Agenten zurück.

        Returns:
            Dict[str, Any]: Die Beobachtung.
        """
        return {
            'total_resources': self.central_planner.environment.total_resources,
            'production_quantity': self.production_quantity,
            'resource_allocation': self.resource_allocation,
            'current_gdp': self.central_planner.environment.get_state().get('current_gdp', 0.0)
        }

    def reset(self) -> None:
        """
        Setzt den Zustand des Agenten zurück.
        """
        self.production_quantity = 0
        self.resource_allocation = {}
        self.total_income = 0.0
        self.reward = 0.0
