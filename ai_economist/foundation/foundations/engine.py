from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np
import threading
import logging
from ai_economist.data_loader import load_gdp_data, get_latest_gdp
import pandas as pd
import os

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentState:
    production_quantity: int = 0
    resource_allocation: Dict[str, Any] = field(default_factory=lambda: {'resource_type': 'Rohstoff', 'amount': 0})

@dataclass
class EconomyEnvironment:
    agents: List['PlannedEconomyAgent'] = field(default_factory=list)
    total_resources: int = 1000  # Beispielwert
    state_lock: threading.Lock = field(default_factory=threading.Lock)
    gdp_data: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    current_date: pd.Timestamp = field(default_factory=lambda: pd.Timestamp('2020-01-01'))
    
    def __post_init__(self):
        # Initialisiere die GDP-Daten
        gdp_file = os.getenv('GDP_DATA_PATH', 'data/gdp_data.csv')
        self.gdp_data = load_gdp_data(gdp_file)
        logger.info("Umgebung erfolgreich initialisiert mit GDP-Daten.")

    def get_state(self) -> Dict[str, Any]:
        """
        Gibt den aktuellen Zustand der Wirtschaft zurück, einschließlich GDP.

        Returns:
            Dict[str, Any]: Der aktuelle Zustand.
        """
        with self.state_lock:
            current_gdp = get_latest_gdp(self.gdp_data, self.current_date)
            state = {
                'total_resources': self.total_resources,
                'agent_states': [agent.get_state() for agent in self.agents],
                'current_gdp': current_gdp
                # Weitere relevante Umweltzustände
            }
        return state

    def apply_central_plan(self, plan: Dict[int, Dict[str, Any]]) -> None:
        """
        Wendet den zentralen Plan auf die Umgebung an.

        Args:
            plan (Dict[int, Dict[str, Any]]): Der Plan für die Agenten.
        """
        threads = []
        for agent_id, action in plan.items():
            agent = self.get_agent_by_id(agent_id)
            if agent:
                thread = threading.Thread(target=agent.apply_action, args=(action,))
                threads.append(thread)
                thread.start()
        for thread in threads:
            thread.join()
        self.update_environment()

    def get_observation(self, agent: 'PlannedEconomyAgent') -> Dict[str, Any]:
        """
        Gibt die Beobachtung für einen bestimmten Agenten zurück.

        Args:
            agent (PlannedEconomyAgent): Der Agent.

        Returns:
            Dict[str, Any]: Die Beobachtung.
        """
        return self.get_agent_observation(agent)

    def update_environment(self) -> None:
        """
        Aktualisiert den Zustand der Umgebung basierend auf den Aktionen der Agenten.
        """
        with self.state_lock:
            for agent in self.agents:
                allocation = agent.resource_allocation.get('amount', 0)
                self.total_resources -= allocation
                # Weitere Aktualisierungen, z.B. GDP-Anpassungen
            # Beispiel: Simuliere GDP-Wachstum
            gdp_growth_rate = 0.02  # 2% Wachstum
            current_gdp = get_latest_gdp(self.gdp_data, self.current_date)
            new_gdp = current_gdp * (1 + gdp_growth_rate)
            new_date = self.current_date + pd.DateOffset(months=1)
            self.current_date = new_date
            # Füge den neuen GDP-Wert hinzu
            self.gdp_data = self.gdp_data.append({'Date': self.current_date, 'GDP': new_gdp}, ignore_index=True)
        logger.info("Umgebung erfolgreich aktualisiert.")

    def get_agent_by_id(self, agent_id: int) -> Optional['PlannedEconomyAgent']:
        """
        Findet einen Agenten anhand seiner ID.

        Args:
            agent_id (int): Die ID des Agenten.

        Returns:
            Optional[PlannedEconomyAgent]: Der Agent oder None.
        """
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None

    def get_state_dimension(self) -> int:
        """
        Definiert die Größe des Zustandsraums für das Modell.

        Returns:
            int: Die Größe des Zustandsraums.
        """
        return len(self.get_state())

    def get_action_dimension(self) -> int:
        """
        Definiert die Größe des Aktionsraums für das Modell.

        Returns:
            int: Die Größe des Aktionsraums.
        """
        return 2  # Beispiel: Produktion und Ressourcenzuweisung

    def reset(self) -> None:
        """
        Setzt die Umgebung zurück.
        """
        with self.state_lock:
            self.total_resources = 1000
            self.current_date = pd.Timestamp('2020-01-01')
            self.gdp_data = load_gdp_data(os.getenv('GDP_DATA_PATH', 'data/gdp_data.csv'))
            for agent in self.agents:
                agent.reset()
            # Weitere Rücksetzungen
        logger.info("Umgebung erfolgreich zurückgesetzt.")

    def collect_experiences(self) -> Dict[str, List[Any]]:
        """
        Sammelt Erfahrungen für das Training.

        Returns:
            Dict[str, List[Any]]: Die gesammelten Erfahrungen.
        """
        with self.state_lock:
            observations = [agent.get_observation() for agent in self.agents]
            actions = [agent.action for agent in self.agents]
            rewards = [agent.reward for agent in self.agents]
            next_observations = [self.get_observation(agent) for agent in self.agents]
        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'next_observations': next_observations
        }

    def step(self, actions: List[Dict[str, Any]]) -> None:
        """
        Führt einen Simulationsschritt durch.

        Args:
            actions (List[Dict[str, Any]]): Die Aktionen der Agenten.
        """
        # Implementiere die Logik für einen Simulationsschritt
        for agent, action in zip(self.agents, actions):
            agent.apply_action(action)
        self.update_environment()

    def get_agent_observation(self, agent: 'PlannedEconomyAgent') -> Dict[str, Any]:
        """
        Implementiert die Beobachtungslogik für einen Agenten.

        Args:
            agent (PlannedEconomyAgent): Der Agent.

        Returns:
            Dict[str, Any]: Die Beobachtung.
        """
        return {
            'total_resources': self.total_resources,
            'production_quantity': agent.production_quantity,
            'resource_allocation': agent.resource_allocation,
            'current_gdp': get_latest_gdp(self.gdp_data, self.current_date)
            # Weitere Beobachtungen
        }
