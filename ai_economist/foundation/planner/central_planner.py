from ai_economist.planner.model import CentralPlannerModel
import numpy as np
from typing import Dict, Any, List, Optional
import threading
import logging
from ai_economist.database import SessionLocal, Experience

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CentralPlanner:
    """
    Zentraler Planer zur Koordination der Wirtschaft und Ressourcenallokation.

    Attributes:
        environment (EconomyEnvironment): Die Wirtschaftsumgebung.
        plan (Dict[int, Dict[str, Any]]): Der aktuelle Plan für die Agenten.
        model (CentralPlannerModel): Das ML-Modell zur Generierung des Plans.
        replay_memory (List[Dict[str, Any]]): Gesammelte Erfahrungen für das Training.
        lock (threading.Lock): Lock zur Thread-Sicherheit.
        strategy (str): Aktuelle Planungsstrategie.
    """

    def __init__(self, environment: 'EconomyEnvironment', strategy: str = 'balance'):
        """
        Initialisiert den zentralen Planer.

        Args:
            environment (EconomyEnvironment): Die Wirtschaftsumgebung.
            strategy (str, optional): Die gewählte Planungsstrategie. Standard ist 'balance'.
        """
        self.environment = environment
        self.plan: Dict[int, Dict[str, Any]] = {}
        self.strategy = strategy
        self.model = self.initialize_model()
        self.replay_memory: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

    def initialize_model(self) -> CentralPlannerModel:
        """
        Initialisiert das ML-Modell basierend auf der gewählten Strategie.

        Returns:
            CentralPlannerModel: Das initialisierte ML-Modell.
        """
        input_size = self.get_state_size()
        output_size = self.get_action_size()
        model = CentralPlannerModel(input_size, output_size, strategy=self.strategy)
        return model

    def get_state_size(self) -> int:
        """
        Definiert die Größe des Zustandsraums.

        Returns:
            int: Die Größe des Zustandsraums.
        """
        state = self.environment.get_state()
        state_vector = self.model.state_to_vector(state)
        return state_vector.shape[0]

    def get_action_size(self) -> int:
        """
        Definiert die Größe des Aktionsraums.

        Returns:
            int: Die Größe des Aktionsraums.
        """
        return self.calculate_total_action_size()

    def calculate_total_action_size(self) -> int:
        """
        Berechnet die gesamte Aktionsgröße basierend auf der Anzahl der Agenten.

        Returns:
            int: Die gesamte Aktionsgröße.
        """
        return len(self.environment.agents) * 2  # Produktion und Ressourcen

    def set_strategy(self, strategy: str) -> None:
        """
        Setzt eine neue Planungsstrategie.

        Args:
            strategy (str): Die neue Strategie.
        """
        with self.lock:
            self.strategy = strategy
            self.model = self.initialize_model()
            logger.info(f"Planungsstrategie auf '{strategy}' gesetzt.")

    def generate_plan(self) -> None:
        """
        Generiert einen zentralen Plan basierend auf dem aktuellen Zustand und der gewählten Strategie.
        """
        state = self.environment.get_state()
        state_vector = self.model.state_to_vector(state)
        try:
            predicted_plan = self.model.predict(state_vector)
            self.plan = self.model.vector_to_plan(predicted_plan)
            logger.info(f"Plan erfolgreich generiert mit Strategie '{self.strategy}'.")
        except Exception as e:
            logger.error(f"Fehler bei der Planerstellung: {e}")
            self.plan = self.default_plan()

    def get_agent_action(self, agent_id: int) -> Dict[str, Any]:
        """
        Gibt die Aktion für einen bestimmten Agenten zurück.

        Args:
            agent_id (int): Die ID des Agenten.

        Returns:
            Dict[str, Any]: Die geplante Aktion.
        """
        with self.lock:
            return self.plan.get(agent_id, self.default_action())

    def train_model(self, experiences: Dict[str, List[Any]]) -> None:
        """
        Trainiert das ML-Modell mit den gesammelten Erfahrungen.

        Args:
            experiences (Dict[str, List[Any]]): Die gesammelten Erfahrungen.
        """
        try:
            # Speichern der Erfahrungen in der Datenbank
            session = SessionLocal()
            for obs, act, rew, next_obs in zip(experiences['observations'], experiences['actions'], experiences['rewards'], experiences['next_observations']):
                experience = Experience(
                    observations=obs,
                    actions=act,
                    rewards=rew,
                    next_observations=next_obs
                )
                session.add(experience)
            session.commit()
            session.close()
            logger.info("Erfahrungen erfolgreich in der Datenbank gespeichert.")

            # Trainiere das Modell mit den neuen Erfahrungen
            self.model.update(experiences)
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Erfahrungen oder Trainieren des Modells: {e}")
            session.rollback()
            session.close()

    def reset(self) -> None:
        """
        Setzt den zentralen Planer zurück.
        """
        with self.lock:
            self.plan = {}
            self.replay_memory = []
            logger.info("Zentraler Planer zurückgesetzt.")

    def collect_experiences(self, observations: List[Any], actions: List[Any], rewards: List[Any], next_observations: List[Any]) -> Dict[str, List[Any]]:
        """
        Sammelt Erfahrungen für das Training.

        Args:
            observations (List[Any]): Beobachtungen vor der Aktion.
            actions (List[Any]): Aktionen der Agenten.
            rewards (List[Any]): Belohnungen der Agenten.
            next_observations (List[Any]): Beobachtungen nach der Aktion.

        Returns:
            Dict[str, List[Any]]: Die gesammelten Erfahrungen.
        """
        experiences = {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'next_observations': next_observations
        }
        return experiences

    def default_plan(self) -> Dict[int, Dict[str, Any]]:
        """
        Definiert einen Standardplan, falls keine spezifischen Aktionen vorhanden sind.

        Returns:
            Dict[int, Dict[str, Any]]: Der Standardplan für alle Agenten.
        """
        return {agent.id: self.model.default_action() for agent in self.environment.agents}
