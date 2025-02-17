import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def generate_data():
    """Génère des données avec un bruit gaussien et des pics aléatoires."""
    base = np.random.normal(loc=0, scale=1)
    if np.random.rand() < 0.02:  # 2% de chance de générer un pic
        return base + np.random.uniform(5, 10)
    return base

class SimulatedMonitor:
    def __init__(self, window_size=100):
        self.model = self._build_lstm_model()
        self.iforest = IsolationForest(contamination=0.05, random_state=42)
        self.window_size = window_size
        self.time_data = deque(maxlen=window_size)
        self.values_data = deque(maxlen=window_size)
        self.anomalies = []
        
        # Configuration initiale du graphique
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self._configure_plot()
        self.line, = self.ax.plot([], [], color='yellow', linewidth=1.5, label='Valeurs')
        self.scatter = self.ax.scatter([], [], color='red', zorder=5, label='Anomalies')

    def _build_lstm_model(self):
        """Construit un modèle LSTM pour la prédiction."""
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(1, 1)))  # LSTM nécessite une entrée 3D
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    def _configure_plot(self):
        """Configure l'apparence du graphique."""
        self.ax.set_facecolor('black')
        for spine in self.ax.spines.values():
            spine.set_color('yellow')
        self.ax.tick_params(colors='yellow')
        self.ax.set_title('Surveillance Simulée - Maintenance Prédictive', color='yellow', pad=20)
        self.ax.set_xlabel('Temps (s)', color='yellow')
        self.ax.set_ylabel('Valeur Capteur', color='yellow')
        self.ax.grid(True, color='gray', linestyle='--', alpha=0.7)
        self.ax.set_xlim(-10, 0)  # Fenêtre temporelle initiale
        self.ax.legend(facecolor='black', labelcolor='yellow')

    def _detect_anomalies(self):
        """Détecte les anomalies avec Isolation Forest et LSTM."""
        if len(self.values_data) < 10:  # Attendre d'avoir suffisamment de données
            return []
        
        data = np.array(self.values_data).reshape(-1, 1)  # Isolation Forest nécessite une entrée 2D
        self.iforest.fit(data)
        preds = self.iforest.predict(data)
        
        # Utilisation du modèle LSTM pour la prédiction
        lstm_input = np.array(self.values_data).reshape(-1, 1, 1)  # Entrée 3D pour LSTM
        lstm_pred = self.model.predict(lstm_input).flatten()  # Aplatir les prédictions LSTM
        
        # Combinaison des résultats
        anomalies = [self.time_data[i] for i, val in enumerate(preds) if val == -1 or abs(lstm_pred[i] - data[i]) > 2]
        return anomalies

    def simulate(self, num_steps=200):
        """Simule des données et met à jour le graphique."""
        for i in range(num_steps):
            new_value = generate_data()
            self.time_data.append(i)  # Utiliser un compteur comme temps simulé
            self.values_data.append(new_value)

            # Détection d'anomalies
            anomaly_times = self._detect_anomalies()
            anomaly_values = [self.values_data[self.time_data.index(t)] for t in anomaly_times]
            
            # Mise à jour des données de la ligne et des anomalies
            self.line.set_data(np.array(self.time_data) - self.time_data[-1], self.values_data)
            self.scatter.set_offsets(np.c_[np.array(anomaly_times) - self.time_data[-1], anomaly_values])
            
            # Ajuster les limites de l'axe en fonction des données
            self.ax.set_xlim(self.time_data[0] - self.time_data[-1] - 5, 5)
            self.ax.relim()
            self.ax.autoscale_view(scaley=False)

        # Sauvegarder le graphique en tant qu'image
        plt.savefig('simulated_monitor.png', bbox_inches='tight', facecolor='black')
        plt.close()

if __name__ == "__main__":
    monitor = SimulatedMonitor(window_size=200)
    monitor.simulate(num_steps=200)
