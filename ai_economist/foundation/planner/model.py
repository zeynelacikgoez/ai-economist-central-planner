import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import logging
import shap
from typing import Dict, Any, List

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CentralPlannerModel:
    """
    ML-Modell für den zentralen Planer zur Vorhersage von Aktionen.

    Attributes:
        model (tf.keras.Model): Das neuronale Netzmodell.
        shap_explainer (shap.Explainer): SHAP-Explainer zur Modellinterpretation.
        strategy (str): Aktuelle Planungsstrategie.
    """

    def __init__(self, input_size: int, output_size: int, strategy: str = 'balance'):
        """
        Initialisiert das ML-Modell.

        Args:
            input_size (int): Die Größe des Eingaberaums.
            output_size (int): Die Größe des Ausgaberaums.
            strategy (str, optional): Die gewählte Planungsstrategie. Standard ist 'balance'.
        """
        self.strategy = strategy
        self.model = self.build_model(input_size, output_size)
        self.initialize_shap()

    def build_model(self, input_size: int, output_size: int) -> tf.keras.Model:
        """
        Baut das neuronale Netzmodell mit einer spezifischen Architektur.

        Args:
            input_size (int): Die Größe des Eingaberaums.
            output_size (int): Die Größe des Ausgaberaums.

        Returns:
            tf.keras.Model: Das erstellte Modell.
        """
        model = models.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_size,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(output_size, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        logger.info("Modell erfolgreich erstellt und kompiliert.")
        return model

    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Macht eine Vorhersage basierend auf dem Zustand.

        Args:
            state (np.ndarray): Der Zustand als Vektor.

        Returns:
            np.ndarray: Die vorhergesagte Aktion.
        """
        try:
            state = state.reshape(1, -1)  # Reshape für das Modell
            prediction = self.model.predict(state)[0]
            return prediction
        except Exception as e:
            logger.error(f"Fehler bei der Vorhersage: {e}")
            return np.zeros(self.model.output_shape[-1])

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 32) -> None:
        """
        Trainiert das Modell mit den gegebenen Daten.

        Args:
            X (np.ndarray): Die Eingabedaten.
            y (np.ndarray): Die Zielwerte.
            epochs (int, optional): Die maximale Anzahl der Epochen. Standard ist 50.
            batch_size (int, optional): Die Batch-Größe. Standard ist 32.
        """
        try:
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
            self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stopping])
            logger.info("Modell erfolgreich trainiert.")
        except Exception as e:
            logger.error(f"Fehler beim Trainieren des Modells: {e}")

    def update(self, experiences: Dict[str, List[Any]]) -> None:
        """
        Aktualisiert das Modell mit neuen Erfahrungen.

        Args:
            experiences (Dict[str, List[Any]]): Die gesammelten Erfahrungen.
        """
        try:
            observations = np.array([self.state_to_vector(obs) for obs in experiences['observations']])
            actions = np.array([self.plan_action_vector(act) for act in experiences['actions']])
            self.train(observations, actions)
            logger.info("Modell erfolgreich mit neuen Erfahrungen aktualisiert.")
        except Exception as e:
            logger.error(f"Fehler beim Aktualisieren des Modells: {e}")

    def initialize_shap(self) -> None:
        """
        Initialisiert den SHAP-Explainer für das Modell.
        """
        try:
            # Initialisiere SHAP explainer, z.B. KernelExplainer
            background = np.zeros((100, self.model.input_shape[1]))  # Größeres Hintergrundset für SHAP
            self.shap_explainer = shap.KernelExplainer(self.model.predict, background)
            logger.info("SHAP Explainer erfolgreich initialisiert.")
        except Exception as e:
            logger.error(f"Fehler beim Initialisieren von SHAP: {e}")

    def explain_prediction(self, state: np.ndarray) -> np.ndarray:
        """
        Erklärt eine Vorhersage mithilfe von SHAP.

        Args:
            state (np.ndarray): Der Zustand als Vektor.

        Returns:
            np.ndarray: Die SHAP-Werte.
        """
        try:
            shap_values = self.shap_explainer.shap_values(state.reshape(1, -1))
            return shap_values
        except Exception as e:
            logger.error(f"Fehler bei der SHAP-Erklärung: {e}")
            return np.zeros(self.model.output_shape[-1])

    def state_to_vector(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Konvertiert den Zustand in einen Vektor.

        Args:
            state (Dict[str, Any]): Der Zustand als Dictionary.

        Returns:
            np.ndarray: Der Zustand als Vektor.
        """
        state_vector = [state.get('total_resources', 0), state.get('current_gdp', 0.0)]
        for agent_state in state.get('agent_states', []):
            state_vector.append(agent_state.get('production_quantity', 0))
            state_vector.append(agent_state.get('resource_allocation', {}).get('amount', 0))
        return np.array(state_vector)

    def vector_to_plan(self, vector: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """
        Konvertiert den Vektor zurück in einen Plan.

        Args:
            vector (np.ndarray): Der vorhergesagte Plan als Vektor.

        Returns:
            Dict[int, Dict[str, Any]]: Der Plan für jeden Agenten.
        """
        plan = {}
        num_agents = int(len(vector) / 2)
        for i in range(num_agents):
            agent_id = i  # Annahme: Agenten IDs sind 0 bis num_agents-1
            production = vector[i * 2]
            resource_amount = vector[i * 2 + 1]
            plan[agent_id] = {
                'production_quantity': int(production),
                'resource_allocation': {'resource_type': 'Rohstoff', 'amount': int(resource_amount)}
            }
        return plan

    def default_action(self) -> Dict[str, Any]:
        """
        Definiert eine Standardaktion.

        Returns:
            Dict[str, Any]: Die Standardaktion.
        """
        return {'production_quantity': 0, 'resource_allocation': {'resource_type': 'Rohstoff', 'amount': 0}}
