import numpy as np
import pandas as pd

# Import only the functions, not running the visualization code
import sys
sys.path.append('.')  # Ensure current directory is in path

# Manually copy the function definitions or restructure your files
# For now, let's assume you've added the if __name__ guards

from Model_1a import total_cost as total_cost_1a
from Model_1a import total_elevator_cost
from Model_1b import total_cost as total_cost_1b

# Generate integer year values from 182 to 5000
T_values = np.arange(182, 5001, 1)

# Calculate costs
model_1a_costs = [total_cost_1a(T) for T in T_values]
model_1b_costs = [total_cost_1b(T) for T in T_values]

elevator_cost = total_elevator_cost()
elevator_costs = [elevator_cost] * len(T_values)

# Create DataFrame
df = pd.DataFrame({
    'T_years': T_values,
    'Model_1a_Rocket_Cost': model_1a_costs,
    'Model_1b_Rocket_Cost': model_1b_costs,
    'Elevator_Cost': elevator_costs,
    'Model_1a_Total': np.array(model_1a_costs) + elevator_cost,
    'Model_1b_Total': np.array(model_1b_costs) + elevator_cost
})

# Save to CSV
df.to_csv('transportation_costs_comparison.csv', index=False)

print("✓ CSV created successfully!")
print(f"Total rows: {len(df)}")
print(df.head())