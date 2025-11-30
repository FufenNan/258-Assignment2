import numpy as np
from sklearn.model_selection import train_test_split
import pickle

with open('training_data.pkl', 'rb') as f:
    training_data = pickle.load(f)

for i, x in enumerate(training_data[:2]):
    print(f"Sample {i}:")
    for k, v in x['user_feature'].items():
        if isinstance(v, np.ndarray):
            print(f"User feature - {k}: array of shape {v.shape}")
        else:
            print(f"User feature - {k}: {v}")
    for k, v in x['game_feature'].items():
        if isinstance(v, np.ndarray):
            print(f"Game feature - {k}: array of shape {v.shape}")
        else:
            print(f"Game feature - {k}: {v}")
    for k, v in x['cross_feature'].items():
        print(f"Cross feature - {k}: {v}")
    print(f"Label: {x['label']}")
    print("-----")

X_user = []
X_game = []
X_cross = []
y = []

for x in training_data:
    # User features
    user_feats = []
    for k, v in x['user_feature'].items():
        if isinstance(v, np.ndarray):
            user_feats.extend(v.tolist())
        else:
            user_feats.append(v)
    X_user.append(user_feats)

    # Game features
    game_feats = []
    for k, v in x['game_feature'].items():
        if isinstance(v, np.ndarray):
            game_feats.extend(v.tolist())
        else:
            game_feats.append(v)
    X_game.append(game_feats)

    # Cross features
    cross_feats = []
    for k, v in x['cross_feature'].items():
        cross_feats.append(v)
    X_cross.append(cross_feats)

    # Label
    y.append(x['label'])

X_user = np.array(X_user, dtype=np.float32)
X_game = np.array(X_game, dtype=np.float32)
X_cross = np.array(X_cross, dtype=np.float32)
y = np.array(y, dtype=np.float32)

X = np.concatenate([X_user, X_game, X_cross], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, shuffle=True
)
print('length of training data:', X_train.shape[0])
print('length of test data:', X_test.shape[0])

# X_train_tensor = torch.from_numpy(X_train)
# y_train_tensor = torch.from_numpy(y_train).unsqueeze(1)  
# X_test_tensor = torch.from_numpy(X_test)
# y_test_tensor = torch.from_numpy(y_test).unsqueeze(1)

# print("Training features shape:", X_train_tensor.shape)
# print("Training labels shape:", y_train_tensor.shape)
# print("Test features shape:", X_test_tensor.shape)
# print("Test labels shape:", y_test_tensor.shape)
