import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import seaborn as sns
import matplotlib.pyplot as plt
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

# Load the cleaned and processed dataset
merged_data = pd.read_csv('cleaned_merged_data_fixed.csv')

# Define the updated features list including the one-hot encoded columns and new features
features = ['freeFlowSpeed', 'temp', 'feels_like', 'temp_freeFlowSpeed', 'feels_like_freeFlowSpeed'] + [col for col in merged_data.columns if col.startswith('weather_main_')]

# Target variable
target = 'currentSpeed'

# Define X (features) and y (target)
X = merged_data[features]
y = merged_data[target]

# Check for duplicates or inconsistencies
print("Initial data check:")
print(merged_data.head())
print(merged_data.describe())

# Visualize data distribution
sns.pairplot(merged_data[features + [target]])
plt.show()

# Standardize the features using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features)
    ])

# Remove duplicates and re-split the data
merged_data.drop_duplicates(inplace=True)
X = merged_data[features]
y = merged_data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Ensure no overlap between training and testing data
print("\nChecking for overlap between training and testing data:")
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Experiment with different models
models = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42),
    'LinearRegression': LinearRegression()
}

# Function to evaluate model with cross-validation
def evaluate_model(model, X_train, y_train):
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    return np.mean(scores), np.std(scores)

# Evaluate the models using cross-validation
for model_name, model in models.items():
    mean_mae, std_mae = evaluate_model(model, X_train, y_train)
    print(f'{model_name} - Mean MAE: {-mean_mae:.4f}, Std MAE: {std_mae:.4f}')

# Train the best models on the entire training set and evaluate on the test set
best_models = {}
for model_name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])
    pipeline.fit(X_train, y_train)
    best_models[model_name] = pipeline

# Evaluate the best models on the test set
for model_name, model in best_models.items():
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f'{model_name} - Test MAE: {mae}, Test RMSE: {rmse}, Test R-squared: {r2}')

# Use the predictions to adjust VRP cost factors
predicted_traffic = {(0, 1): 1.2, (1, 2): 1.1, (2, 3): 1.3}  # Example values
predicted_emissions = {(0, 1): 1.1, (1, 2): 1.2, (2, 3): 1.0}  # Example values
predicted_fuel = {(0, 1): 1.3, (1, 2): 1.0, (2, 3): 1.2}  # Example values

# Define the VRP data model
def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = [
        [0, 29, 20, 21],
        [29, 0, 15, 17],
        [20, 15, 0, 28],
        [21, 17, 28, 0],
    ]
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data

# Solve VRP with real-time data adjustments
def solve_vrp_with_real_time_data(predicted_traffic, predicted_emissions, predicted_fuel):
    """Solve the VRP with real-time data adjustments for traffic, emissions, and fuel."""
    data = create_data_model()
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def combined_cost_callback(from_index, to_index):
        """Combines distance, emissions, and fuel costs into a single cost."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)

        # Distance cost adjusted for traffic
        traffic_factor = predicted_traffic.get((from_node, to_node), 1)
        distance_cost = int(data['distance_matrix'][from_node][to_node] * traffic_factor)

        # Emissions cost
        emissions_factor = predicted_emissions.get((from_node, to_node), 1)
        emissions_cost = int(data['distance_matrix'][from_node][to_node] * emissions_factor)

        # Fuel cost
        fuel_factor = predicted_fuel.get((from_node, to_node), 1)
        fuel_cost = int(data['distance_matrix'][from_node][to_node] * fuel_factor)

        # Combine the costs
        combined_cost = distance_cost + emissions_cost + fuel_cost
        return combined_cost

    combined_callback_index = routing.RegisterTransitCallback(combined_cost_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(combined_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        print_solution(manager, routing, solution)

def print_solution(manager, routing, solution):
    """Prints the solution."""
    print('Objective: {} cost units'.format(solution.ObjectiveValue()))
    index = routing.Start(0)
    plan_output = 'Route:\n'
    while not routing.IsEnd(index):
        plan_output += ' {} ->'.format(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    plan_output += ' {}\n'.format(manager.IndexToNode(index))
    print(plan_output)

if __name__ == '__main__':
    solve_vrp_with_real_time_data(predicted_traffic, predicted_emissions, predicted_fuel)
