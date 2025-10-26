"""
Linear Regression with Stochastic Gradient Descent (SGD) - Professional Version
Complete ML Pipeline Implementation with Professional Terminal Output (Dashboard Edition)

This file implements a complete ML pipeline following the standard sequence:
1. Problem Definition
2. Data Collection  
3. Data Preprocessing
4. Data Exploration & Visualization
5. Model Selection
6. Model Training
7. Model Evaluation
8. Model Optimization
9. Model Deployment
"""

import csv
import random
import math


# FIX FOR WINDOWS: Set the matplotlib backend to ensure plots are displayed

import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend for interactive plots

import matplotlib.pyplot as plt
import numpy as np


# CONFIGURATION SETTINGS - Edit these values as needed

LEARNING_RATE = 0.01          # SGD learning rate
N_ITERATIONS = 1000           # Number of training epochs
TEST_SIZE = 0.2               # Train/test split ratio (0.2 = 20% test)
RANDOM_SEED = 42              # For reproducible results
SHOW_PLOTS = True             # Set to True to enable visualizations
PRINT_PROGRESS_EVERY = 100    # Print training progress every N iterations


# 1. PROBLEM DEFINITION

"""
Problem: Predict a target variable using 3 input features
Dataset: MultipleLR-Dataset.csv with 25 samples, 4 columns (3 features + 1 target)
Goal: Implement Linear Regression using Stochastic Gradient Descent from scratch
Requirements: No external ML libraries, professional interface, comprehensive analysis
"""

# ANSI Color Codes for terminal formatting
class Colors:
    RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = '\033[91m', '\033[92m', '\033[93m', '\033[94m', '\033[95m', '\033[96m', '\033[97m'
    BOLD, RESET = '\033[1m', '\033[0m'
    HEADER, SUCCESS, WARNING, ERROR, INFO, HIGHLIGHT = BOLD + CYAN, BOLD + GREEN, BOLD + YELLOW, BOLD + RED, BOLD + BLUE, BOLD + MAGENTA


# 2. DATA COLLECTION

def load_csv(filename):
    """Load data from CSV file"""
    feature_data, target_data = [], []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 4:
                feature_data.append([float(val) for val in row[:3]])
                target_data.append(float(row[3]))
    return feature_data, target_data


# 3. DATA PREPROCESSING

def normalize_features(feature_data):
    """Normalize input features to [0, 1] range"""
    number_of_features = len(feature_data[0])
    normalized_feature_data = []
    feature_minimums = [min(col) for col in zip(*feature_data)]
    feature_maximums = [max(col) for col in zip(*feature_data)]
    
    for row in feature_data:
        normalized_row = [(row[j] - feature_minimums[j]) / (feature_maximums[j] - feature_minimums[j]) if (feature_maximums[j] - feature_minimums[j]) != 0 else 0 for j in range(number_of_features)]
        normalized_feature_data.append(normalized_row)
    
    return normalized_feature_data, feature_minimums, feature_maximums

def train_test_split(feature_data, target_data, test_size=TEST_SIZE, random_seed=RANDOM_SEED):
    """Split data into training and testing sets"""
    random.seed(random_seed)
    number_of_samples = len(feature_data)
    indices = list(range(number_of_samples))
    random.shuffle(indices)
    
    split_index = int(number_of_samples * (1 - test_size))
    training_set_indices, testing_set_indices = indices[:split_index], indices[split_index:]
    
    feature_training_set, feature_testing_set = [feature_data[i] for i in training_set_indices], [feature_data[i] for i in testing_set_indices]
    target_training_set, target_testing_set = [target_data[i] for i in training_set_indices], [target_data[i] for i in testing_set_indices]
    
    return feature_training_set, feature_testing_set, target_training_set, target_testing_set


# 5. MODEL SELECTION

class LinearRegressionSGD:
    def __init__(self, learning_rate=LEARNING_RATE, n_iterations=N_ITERATIONS, random_seed=RANDOM_SEED):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_seed = random_seed
        self.weights = None
        self.bias = None
        self.losses = []
    
    def fit(self, feature_data, target_data):
        random.seed(self.random_seed)
        number_of_samples, number_of_features = len(feature_data), len(feature_data[0])
        self.weights = [random.uniform(-0.1, 0.1) for _ in range(number_of_features)]
        self.bias = 0.0
        
        for iteration in range(self.n_iterations):
            total_loss = 0
            indices = list(range(number_of_samples))
            random.shuffle(indices)
            
            for idx in indices:
                predicted_target_value = self._predict_single(feature_data[idx])
                error = predicted_target_value - target_data[idx]
                total_loss += error ** 2
                
                for j in range(number_of_features):
                    self.weights[j] -= self.learning_rate * error * feature_data[idx][j]
                self.bias -= self.learning_rate * error
            
            avg_loss = total_loss / number_of_samples
            self.losses.append(avg_loss)
            
            if (iteration + 1) % PRINT_PROGRESS_EVERY == 0:
                print(f"{Colors.INFO}Iteration {iteration + 1}/{self.n_iterations}{Colors.RESET}, {Colors.SUCCESS}Loss: {avg_loss:.4f}{Colors.RESET}")
    
    def _predict_single(self, feature_vector):
        return self.bias + sum(self.weights[i] * feature_vector[i] for i in range(len(feature_vector)))
    
    def predict(self, feature_data):
        return [self._predict_single(feature_vector) for feature_vector in feature_data]
    
    def score(self, feature_data, target_data):
        predicted_target_values = self.predict(feature_data)
        mean_of_target_values = sum(target_data) / len(target_data)
        total_sum_of_squares = sum((y_i - mean_of_target_values) ** 2 for y_i in target_data)
        residual_sum_of_squares = sum((y_i - y_pred_i) ** 2 for y_i, y_pred_i in zip(target_data, predicted_target_values))
        return 1 - (residual_sum_of_squares / total_sum_of_squares) if total_sum_of_squares != 0 else 0
    
    def mean_squared_error(self, feature_data, target_data):
        predicted_target_values = self.predict(feature_data)
        return sum((y_i - y_pred_i) ** 2 for y_i, y_pred_i in zip(target_data, predicted_target_values)) / len(target_data)
    
    def mean_absolute_error(self, feature_data, target_data):
        predicted_target_values = self.predict(feature_data)
        return sum(abs(y_i - y_pred_i) for y_i, y_pred_i in zip(target_data, predicted_target_values)) / len(target_data)


# 6. MODEL TRAINING & EVALUATION

def train_model(feature_training_set, target_training_set):
    print(f"\n{Colors.INFO}[4]{Colors.RESET} {Colors.BOLD}Training Linear Regression model...{Colors.RESET}")
    model = LinearRegressionSGD()
    model.fit(feature_training_set, target_training_set)
    print(f"    {Colors.SUCCESS}✓ Model training completed!{Colors.RESET}")
    return model

def evaluate_model(model, normalized_feature_training_set, target_training_set, normalized_feature_testing_set, target_testing_set):
    print(f"\n{Colors.HEADER}{'=' * 60}\n{Colors.HEADER}{Colors.BOLD}RESULTS\n{Colors.HEADER}{'=' * 60}{Colors.RESET}")
    
    print(f"\n{Colors.HIGHLIGHT}Model Parameters{Colors.RESET}")
    print(f"  {Colors.BOLD}Weights:{Colors.RESET} {Colors.CYAN}{[f'{w:.4f}' for w in model.weights]}{Colors.RESET}")
    print(f"  {Colors.BOLD}Bias:{Colors.RESET} {Colors.CYAN}{model.bias:.4f}{Colors.RESET}")
    
    training_r2_score, training_mean_squared_error, training_mean_absolute_error = model.score(normalized_feature_training_set, target_training_set), model.mean_squared_error(normalized_feature_training_set, target_training_set), model.mean_absolute_error(normalized_feature_training_set, target_training_set)
    testing_r2_score, testing_mean_squared_error, testing_mean_absolute_error = model.score(normalized_feature_testing_set, target_testing_set), model.mean_squared_error(normalized_feature_testing_set, target_testing_set), model.mean_absolute_error(normalized_feature_testing_set, target_testing_set)
    
    print(f"\n{Colors.HIGHLIGHT}Training Performance{Colors.RESET}")
    print(f"  {Colors.BOLD}R² Score:{Colors.RESET} {Colors.SUCCESS}{training_r2_score:.4f}{Colors.RESET}, {Colors.BOLD}MSE:{Colors.RESET} {Colors.YELLOW}{training_mean_squared_error:.4f}{Colors.RESET}, {Colors.BOLD}MAE:{Colors.RESET} {Colors.YELLOW}{training_mean_absolute_error:.4f}{Colors.RESET}")
    
    print(f"\n{Colors.HIGHLIGHT}Testing Performance{Colors.RESET}")
    print(f"  {Colors.BOLD}R² Score:{Colors.RESET} {Colors.SUCCESS}{testing_r2_score:.4f}{Colors.RESET}, {Colors.BOLD}MSE:{Colors.RESET} {Colors.YELLOW}{testing_mean_squared_error:.4f}{Colors.RESET}, {Colors.BOLD}MAE:{Colors.RESET} {Colors.YELLOW}{testing_mean_absolute_error:.4f}{Colors.RESET}")
    
    print(f"\n{Colors.HIGHLIGHT}Sample Predictions on Test Set{Colors.RESET}")
    predicted_target_testing_values = model.predict(normalized_feature_testing_set)
    for i in range(min(5, len(normalized_feature_testing_set))):
        error = abs(target_testing_set[i] - predicted_target_testing_values[i])
        error_color = Colors.SUCCESS if error < 1.0 else Colors.WARNING if error < 2.0 else Colors.ERROR
        print(f"  {Colors.BOLD}Sample {i+1}:{Colors.RESET} {Colors.CYAN}Actual = {target_testing_set[i]:.2f}{Colors.RESET}, {Colors.MAGENTA}Predicted = {predicted_target_testing_values[i]:.2f}{Colors.RESET}, {error_color}Error = {error:.2f}{Colors.RESET}")


# 8. MODEL OPTIMIZATION

def optimize_model(normalized_feature_training_set, target_training_set, normalized_feature_testing_set, target_testing_set):
    print(f"\n{Colors.HIGHLIGHT}Model Optimization{Colors.RESET}")
    best_r2_score, best_learning_rate = -float('inf'), None
    
    for learning_rate_value in [0.001, 0.01, 0.1]:
        model = LinearRegressionSGD(learning_rate=learning_rate_value, n_iterations=500, random_seed=42)
        model.fit(normalized_feature_training_set, target_training_set)
        r2_score = model.score(normalized_feature_testing_set, target_testing_set)
        print(f"  {Colors.BOLD}Learning Rate: {learning_rate_value}{Colors.RESET}, {Colors.CYAN}Test R²: {r2_score:.4f}{Colors.RESET}")
        if r2_score > best_r2_score: best_r2_score, best_learning_rate = r2_score, learning_rate_value
            
    print(f"  {Colors.SUCCESS}✓ Best Learning Rate: {best_learning_rate} (R²: {best_r2_score:.4f}){Colors.RESET}")
    return best_learning_rate


# 9. MODEL DEPLOYMENT (DASHBOARD VISUALIZATION)

def create_dashboard(model, normalized_feature_training_set, target_training_set, normalized_feature_testing_set, target_testing_set):
    """Create a single, comprehensive visualization dashboard."""
    if not SHOW_PLOTS: return
        
    print(f"\n{Colors.HIGHLIGHT}Creating Full Analysis Dashboard...{Colors.RESET}")
    
    try:
        predicted_target_training_values = model.predict(normalized_feature_training_set)
        predicted_target_testing_values = model.predict(normalized_feature_testing_set)
        
        # Create a 3x3 grid for the plots
        fig, axs = plt.subplots(3, 3, figsize=(22, 18))
        fig.suptitle('Linear Regression with SGD - Complete Analysis Dashboard', fontsize=20, fontweight='bold')

        # 1. Training Loss Convergence
        axs[0, 0].plot(model.losses, color='#2E86AB', linewidth=2)
        axs[0, 0].set_title('1. Training Loss Convergence', fontsize=14, fontweight='bold')
        axs[0, 0].set_xlabel('Iteration', fontweight='bold')
        axs[0, 0].set_ylabel('Loss (MSE)', fontweight='bold')
        axs[0, 0].grid(True, alpha=0.3)

        # 2. Training Set: Predictions vs Actual
        axs[0, 1].scatter(target_training_set, predicted_target_training_values, alpha=0.7, color='#2E86AB')
        min_val_train = min(min(target_training_set), min(predicted_target_training_values))
        max_val_train = max(max(target_training_set), max(predicted_target_training_values))
        axs[0, 1].plot([min_val_train, max_val_train], [min_val_train, max_val_train], 'r--', lw=2)
        axs[0, 1].set_title('2. Training Set: Predictions vs Actual', fontsize=14, fontweight='bold')
        axs[0, 1].set_xlabel('Actual Values', fontweight='bold')
        axs[0, 1].set_ylabel('Predicted Values', fontweight='bold')
        axs[0, 1].grid(True, alpha=0.3)

        # 3. Test Set: Predictions vs Actual
        axs[0, 2].scatter(target_testing_set, predicted_target_testing_values, alpha=0.8, color='#E63946')
        min_val_test = min(min(target_testing_set), min(predicted_target_testing_values))
        max_val_test = max(max(target_testing_set), max(predicted_target_testing_values))
        axs[0, 2].plot([min_val_test, max_val_test], [min_val_test, max_val_test], 'r--', lw=2)
        axs[0, 2].set_title('3. Test Set: Predictions vs Actual', fontsize=14, fontweight='bold')
        axs[0, 2].set_xlabel('Actual Values', fontweight='bold')
        axs[0, 2].set_ylabel('Predicted Values', fontweight='bold')
        axs[0, 2].grid(True, alpha=0.3)
        
        # 4. Residual Analysis
        residuals = [actual - pred for actual, pred in zip(target_testing_set, predicted_target_testing_values)]
        axs[1, 0].scatter(predicted_target_testing_values, residuals, alpha=0.7, color='#F18F01')
        axs[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=2)
        axs[1, 0].set_title('4. Residual Analysis', fontsize=14, fontweight='bold')
        axs[1, 0].set_xlabel('Predicted Values', fontweight='bold')
        axs[1, 0].set_ylabel('Residuals', fontweight='bold')
        axs[1, 0].grid(True, alpha=0.3)
        
        # 5. Feature Importance
        feature_names = ['Feature 1', 'Feature 2', 'Feature 3']
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        axs[1, 1].bar(feature_names, model.weights, color=colors, alpha=0.8)
        axs[1, 1].set_title('5. Feature Importance (Weights)', fontsize=14, fontweight='bold')
        axs[1, 1].set_ylabel('Weight Value', fontweight='bold')
        axs[1, 1].grid(True, alpha=0.3, axis='y')
        
        # 6. Error Distribution
        errors = [abs(actual - pred) for actual, pred in zip(target_testing_set, predicted_target_testing_values)]
        axs[1, 2].hist(errors, bins=8, alpha=0.7, color='#A23B72', edgecolor='white')
        axs[1, 2].axvline(np.mean(errors), color='red', linestyle='--', lw=2, label=f'Mean Error: {np.mean(errors):.2f}')
        axs[1, 2].set_title('6. Error Distribution (Test Set)', fontsize=14, fontweight='bold')
        axs[1, 2].set_xlabel('Absolute Error', fontweight='bold')
        axs[1, 2].set_ylabel('Frequency', fontweight='bold')
        axs[1, 2].legend()
        axs[1, 2].grid(True, alpha=0.3, axis='y')
        
        # 7. Performance Metrics Comparison
        metrics = ['R² Score', 'MSE', 'MAE']
        training_r2_score = model.score(normalized_feature_training_set, target_training_set)
        testing_r2_score = model.score(normalized_feature_testing_set, target_testing_set)
        training_mean_squared_error = model.mean_squared_error(normalized_feature_training_set, target_training_set)
        testing_mean_squared_error = model.mean_squared_error(normalized_feature_testing_set, target_testing_set)
        training_mean_absolute_error = model.mean_absolute_error(normalized_feature_training_set, target_training_set)
        testing_mean_absolute_error = model.mean_absolute_error(normalized_feature_testing_set, target_testing_set)
        train_values = [training_r2_score, training_mean_squared_error, training_mean_absolute_error]
        test_values = [testing_r2_score, testing_mean_squared_error, testing_mean_absolute_error]
        x = np.arange(len(metrics))
        width = 0.35
        axs[2, 0].bar(x - width/2, train_values, width, label='Training Set', color='#2E86AB')
        axs[2, 0].bar(x + width/2, test_values, width, label='Test Set', color='#E63946')
        axs[2, 0].set_title('7. Performance Metrics: Train vs Test', fontsize=14, fontweight='bold')
        axs[2, 0].set_ylabel('Values', fontweight='bold')
        axs[2, 0].set_xticks(x)
        axs[2, 0].set_xticklabels(metrics)
        axs[2, 0].legend()
        axs[2, 0].grid(True, alpha=0.3, axis='y')

        # Hide unused subplots for a cleaner look
        fig.delaxes(axs[2, 1])
        fig.delaxes(axs[2, 2])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
        plt.show()
        
        print(f"  {Colors.SUCCESS}✓ Dashboard displayed successfully!{Colors.RESET}")
        
    except Exception as e:
        print(f"  {Colors.ERROR}✗ Visualization error: {e}{Colors.RESET}")

def main():
    """Main function executing the complete ML pipeline"""
    print(f"{Colors.HEADER}{'=' * 60}\n{Colors.HEADER}{Colors.BOLD}Linear Regression with SGD - Complete ML Pipeline\n{Colors.HEADER}{'=' * 60}{Colors.RESET}")
    
    # Data Collection
    print(f"\n{Colors.INFO}[1]{Colors.RESET} {Colors.BOLD}Data Collection...{Colors.RESET}")
    try:
        feature_data, target_data = load_csv("MultipleLR-Dataset.csv")
        print(f"    {Colors.SUCCESS}[OK] Loaded {len(feature_data)} samples with {len(feature_data[0])} features (4 columns total){Colors.RESET}")
    except FileNotFoundError:
        print(f"    {Colors.ERROR}✗ ERROR: 'MultipleLR-Dataset.csv' not found!{Colors.RESET}")
        return
    
    # Data Preprocessing
    print(f"\n{Colors.INFO}[2]{Colors.RESET} {Colors.BOLD}Data Preprocessing...{Colors.RESET}")
    feature_training_set, feature_testing_set, target_training_set, target_testing_set = train_test_split(feature_data, target_data)
    print(f"    {Colors.SUCCESS}✓ Data split: {len(feature_training_set)} training samples, {len(feature_testing_set)} testing samples.{Colors.RESET}")
    
    normalized_feature_training_set, feature_minimums, feature_maximums = normalize_features(feature_training_set)
    normalized_feature_testing_set = [ [(row[j] - feature_minimums[j]) / (feature_maximums[j] - feature_minimums[j]) if (feature_maximums[j] - feature_minimums[j]) != 0 else 0 for j in range(len(row))] for row in feature_testing_set]
    print(f"    {Colors.SUCCESS}✓ Features normalized.{Colors.RESET}")
    
    # Model Training, Evaluation, and Optimization
    model = train_model(normalized_feature_training_set, target_training_set)
    evaluate_model(model, normalized_feature_training_set, target_training_set, normalized_feature_testing_set, target_testing_set)
    best_learning_rate = optimize_model(normalized_feature_training_set, target_training_set, normalized_feature_testing_set, target_testing_set)
    
    # Final Model Deployment (Visualization Dashboard)
    print(f"\n{Colors.INFO}[5]{Colors.RESET} {Colors.BOLD}Generating Final Dashboard with optimized learning rate ({best_learning_rate})...{Colors.RESET}")
    final_model = LinearRegressionSGD(learning_rate=best_learning_rate)
    final_model.fit(normalized_feature_training_set, target_training_set)
    create_dashboard(final_model, normalized_feature_training_set, target_training_set, normalized_feature_testing_set, target_testing_set)
    
    print(f"\n{Colors.HEADER}{'=' * 60}\n{Colors.SUCCESS}{Colors.BOLD}ML Pipeline completed successfully!\n{Colors.HEADER}{'=' * 60}{Colors.RESET}")

if __name__ == "__main__":
    main()