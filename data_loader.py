from flamby.datasets.fed_heart_disease import FedHeartDisease
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_fed_heart_disease():
    dataset = FedHeartDisease()
    clients_data = dataset.get_train_clients()
    clients_x = []
    clients_y = []
    scaler = StandardScaler()

    # Fit scaler on all client data combined
    all_x = np.vstack([client[0] for client in clients_data])
    scaler.fit(all_x)

    for x, y in clients_data:
        x_scaled = scaler.transform(x)
        clients_x.append(x_scaled)
        clients_y.append(y)

    test_data = dataset.get_test_data()
    test_x = scaler.transform(test_data[0])
    test_y = test_data[1]

    return clients_x, clients_y, test_x, test_y
