import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Load the training data
train_df = pd.read_csv('./CMaps/train_FD001.txt', sep=' ', header=None)
# Drop the last two columns (NaN values due to extra spaces)
train_df.drop(columns=[26, 27], inplace=True)
# Assign column names
columns = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + \
          [f'sensor{i}' for i in range(1, 22)]
train_df.columns = columns

# Compute RUL for each cycle
max_cycles = train_df.groupby('engine_id')['cycle'].max()
def compute_rul(row):
    max_cycle = max_cycles[row['engine_id']]
    rul = max_cycle - row['cycle']
    return min(rul, 130)  # Cap RUL at 130 cycles
train_df['RUL'] = train_df.apply(compute_rul, axis=1)

# Define features (exclude engine_id, cycle, and RUL)
features = ['setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)]
X = train_df[features]
y = train_df['RUL']

# Split data by engines to avoid leakage
engine_ids = train_df['engine_id'].unique()
train_engines, val_engines = train_test_split(engine_ids, test_size=0.2, random_state=42)
train_data = train_df[train_df['engine_id'].isin(train_engines)]
val_data = train_df[train_df['engine_id'].isin(val_engines)]

# Extract features and targets
X_train = train_data[features]
y_train = train_data['RUL']
X_val = val_data[features]
y_val = val_data['RUL']

# Normalize the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

print("Data processing complete:")
print(f"Training samples: {X_train_tensor.shape[0]}, Validation samples: {X_val_tensor.shape[0]}")
print(f"Number of features: {X_train_tensor.shape[1]}")



# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Training function
def train_model(model, optimizer, X_train, y_train, X_val, y_val, epochs=100):
    criterion = nn.MSELoss()
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
        
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    
    return train_losses, val_losses

# Model parameters
input_size = X_train_tensor.shape[1]  # Number of features (24)
hidden_size = 64  # Example hidden layer size
output_size = 1   # RUL prediction

# Dictionary of optimizers to compare
optimizers = {
    'AdamW': lambda params: optim.AdamW(params, lr=0.001),
    'AdamW_no_momentum': lambda params: optim.AdamW(params, lr=0.001, betas=(0, 0.999)),
    'SGD_momentum': lambda params: optim.SGD(params, lr=0.001, momentum=0.9),
    'SGD': lambda params: optim.SGD(params, lr=0.001)
}

# Train and evaluate each optimizer
results = {}
for opt_name, opt_func in optimizers.items():
    print(f"\nTraining with {opt_name}")
    # Initialize a fresh model for each optimizer
    model = MLP(input_size, hidden_size, output_size)
    optimizer = opt_func(model.parameters())
    train_losses, val_losses = train_model(
        model, optimizer, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, epochs=100
    )
    results[opt_name] = {'train_losses': train_losses, 'val_losses': val_losses}
    print(f"Final Validation Loss with {opt_name}: {val_losses[-1]:.4f}")
    
    

# --- Plotting the Results ---
print("\nPlotting training and validation losses...")

plt.figure(figsize=(12, 5))

# Plot Training Losses
plt.subplot(1, 2, 1)
for opt_name, data in results.items():
    plt.plot(data['train_losses'], label=f'{opt_name}')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)

# Plot Validation Losses
plt.subplot(1, 2, 2)
for opt_name, data in results.items():
    plt.plot(data['val_losses'], label=f'{opt_name}')
plt.title('Validation Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
plt.savefig('training_validation_losses.png', dpi=300)
# --- Feature Importance Analysis (Basic - based on first layer weights) ---
print("\nPerforming basic feature importance analysis...")

# Find the optimizer with the best final validation loss
best_opt_name = min(results, key=lambda k: results[k]['val_losses'][-1])
print(f"Analyzing feature importance based on the model trained with: {best_opt_name}")

# Re-initialize and train the best model again to get access to its final state
# (Alternatively, you could save the model state during the loop)
best_model = MLP(input_size, hidden_size, output_size)
best_optimizer = optimizers[best_opt_name](best_model.parameters())
# Retrain (or load if saved) - using the same data splits
train_model(best_model, best_optimizer, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, epochs=100)

# Get the weights of the first layer
best_model.eval() # Set model to evaluation mode
first_layer_weights = best_model.fc1.weight.data.numpy()

# Calculate the average absolute weight for each input feature
# This gives a rough idea of how much each feature contributes *directly* to the hidden layer activations
feature_importance = np.mean(np.abs(first_layer_weights), axis=0)

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': features, # Use the 'features' list defined during preprocessing
    'Importance': feature_importance
})

# Sort features by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

print("\nFeature Importance (based on average absolute weight in the first layer):")
print(importance_df)

# Plot feature importance
plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Average Absolute Weight')
plt.ylabel('Feature')
plt.title(f'Basic Feature Importance (MLP First Layer - Trained with {best_opt_name})')
plt.gca().invert_yaxis() # Display most important features at the top
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
plt.savefig('feature_importance.png', dpi=300)

# print("\n--- Important Note on Feature Importance ---")
# print("The feature importance shown above is based solely on the average absolute weights of the *first layer* of the MLP.")
# print("While simple to calculate, this method has limitations for multi-layer neural networks:")
# print(" - It doesn't fully capture complex interactions between features in deeper layers.")
# print(" - Non-linear activations (like ReLU) affect how weights translate to importance.")
# print("For more robust feature importance analysis in neural networks, consider methods like:")
# print(" - Permutation Importance: Measures performance drop when a feature is shuffled.")
# print(" - SHAP (SHapley Additive exPlanations) or Integrated Gradients: More advanced techniques to attribute predictions to features.")
# print("--------------------------------------------")