import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from typing import Tuple, List
from sklearn.metrics import roc_auc_score, auc , roc_curve

'''
General Instructions:

1. Do not use any additional libraries. Your code will be tested in a pre-built environment with only 
the library above available.

2. You are expected to fill in the skeleton code precisely as per provided. On top of skeleton code given,
you may write whatever deemed necessary to complete the assignment. For example, you may define additional 
default arguments, class parameters, or methods to help you complete the assignment.

3. Some initial steps or definition are given, aiming to help you getting started. As long as you follow 
the argument and return type, you are free to change them as you see fit.

4. Your code should be free of compilation errors. Compilation errors will result in 0 marks.
'''

class DataProcessor:
    def __init__(self, data_root: str):
        """ Initialize data processor with paths to train and test data.
        
        Args:
            data_root: root path to data directory
        """
        self.data_root = data_root
        self.train_mean = None
        self.train_std = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ Load training and test data from CSV files.
        
        Returns:
            Tuple containing training and test dataframes
        """
        # TODO: Implement data loading
        train_data = pd.read_csv(f"{self.data_root}/data_train_25s.csv")
        test_data = pd.read_csv(f"{self.data_root}/data_test_25s.csv")
        return train_data, test_data
        
    def check_missing_values(self, data: pd.DataFrame) -> int:
        """ Count number of missing values in dataset.
        
        Args:
            data: Input dataframe
            
        Returns:
            Number of missing values
        """
        # TODO: Implement missing value check
        return data.isnull().sum().sum()
        
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Remove rows with missing values.
        
        Args:
            data: Input dataframe
            
        Returns:
            Cleaned dataframe
        """
        # TODO: Implement data cleaning
        return data.dropna()
        
    def extract_features_labels(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """ Extract features and labels from dataframe, convert to numpy arrays.
        
        Args:
            data: Input dataframe
            
        Returns:
            Tuple of feature matrix X and label vector y
        """
        # TODO: Implement feature/label extraction
        X = data.drop(columns=['PT08.S1(CO)']).values
        y = data['PT08.S1(CO)'].values
        if self.train_mean is None:
            self.train_mean = X.mean(axis=0)
            self.train_std = X.std(axis=0)
        X = (X - self.train_mean) / self.train_std

        return X, y
    
class LinearRegression:
    def __init__(self, learning_rate=0.15, max_iter=10000):
        """ Initialize linear regression model.
        
        Args:
            learning_rate: Learning rate for gradient descent
            max_iter: Maximum number of iterations
            l2_lambda: L2 regularization strength
        """
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        print(f"LR {learning_rate} | Epochs {max_iter}")
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        """ Train linear regression model.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            List of loss values
        """
        # TODO: Implement linear regression training
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Initialize with zeros
        self.bias = 0
        losses = []
        
        for _ in range(self.max_iter):

            y_pred = self.predict(X)
            
            # loss
            loss = self.criterion(y, y_pred)
            losses.append(loss)
            
            # Compute gradients
            dw = (2/n_samples) * np.dot(X.T, (y_pred - y))  
            db = (2/n_samples) * np.sum(y_pred - y)  

            #  early stopping
            if len(losses) > 10 and abs(losses[-1] - losses[-2]) < 1e-8:
                print("Stopped at ", _)
                break
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Make predictions with trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted values
        """
        # TODO: Implement linear regression prediction
        return np.dot(X, self.weights) + self.bias

    def criterion(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ Calculate MSE loss.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Loss value
        """
        # TODO: Implement loss function
        return np.mean((y_true - y_pred) ** 2)

    def metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ Calculate RMSE.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Metric value
        """
        # TODO: Implement RMSE calculation
        return np.sqrt(self.criterion(y_true, y_pred))

class LogisticRegression:
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 5000):

        """ Initialize logistic regression model.
        
        Args:
            learning_rate: Learning rate for gradient descent
            max_iter: Maximum number of iterations
        """
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        """ Train logistic regression model with normalization and L2 regularization.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            List of loss values
        """
        # TODO: Implement logistic regression training
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        loss_values = []

        # y = np.array(y).reshape(-1)  # Ensure y is 1D array

        for _ in range(self.max_iter):
            y_predicted = self.predict_proba(X)
            loss = self.criterion(y, y_predicted)
            loss_values.append(loss)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return loss_values
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """ Calculate prediction probabilities using normalized features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Prediction probabilities
        """
        # TODO: Implement logistic regression prediction probabilities
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Make predictions with trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted values
        """
        # TODO: Implement logistic regression prediction
        return (self.predict_proba(X) >= 0.5).astype(int)

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def criterion(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ Calculate BCE loss.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Loss value
        """
        # TODO: Implement loss function
        epsilon = 1e-15
        return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
    
    def F1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ Calculate F1 score with handling of edge cases.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            F1 score (between 0 and 1), or 0.0 for edge cases
        """
        # TODO: Implement F1 score calculation
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        
        if y_true.size == 0 or y_pred.size == 0:
            return 0.0
        
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = true_positives / max(true_positives + false_positives, 1e-15)
        recall = true_positives / max(true_positives + false_negatives, 1e-15)
        
        if precision == 0 and recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / max(precision + recall, 1e-15)
        return float(f1)

    def label_binarize(self, y: np.ndarray) -> np.ndarray:
        """ Binarize labels for binary classification.
        
        Args:
            y: Target vector
            
        Returns:
            Binarized labels
        """
        # TODO: Implement label binarization
        if y is None:
            return None
        return (y > 1000).astype(int)

    def get_auroc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ Calculate AUROC score.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            
        Returns:
            AUROC score (between 0 and 1)
        """
        # TODO: Implement AUROC calculation
        auc_score = auc_score = roc_auc_score(y_true, y_pred)
        print("AUC:", auc_score)
        return auc_score

    def metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ Calculate AUROC.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            AUROC score
        """
        # TODO: Implement AUROC calculation
        return self.get_auroc(y_true, y_pred)

class ModelEvaluator:
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """ Initialize evaluator with number of CV splits.
        
        Args:
            n_splits: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
    def cross_validation(self, model, X: np.ndarray, y: np.ndarray) -> List[float]:
        """ Perform cross-validation
        
        Args:
            model: Model to be evaluated
            X: Feature matrix
            y: Target vector
            
        Returns:
            List of metric scores
        """
        # TODO: Implement cross-validation
        scores = []
        
        for train_idx, val_idx in self.kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            if isinstance(model, LogisticRegression):
                y_binary = model.label_binarize(y_train)
                y_val_binary = model.label_binarize(y_val)
                model.fit(X_train, y_binary)
                y_pred = model.predict(X_val)
                y_pred_probs = model.predict_proba(X_val)
                auroc = model.get_auroc(y_val_binary, y_pred_probs)
                f1 = model.F1_score(y_val_binary, y_pred)
                scores.append((auroc, f1))
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                metric_score = model.metric(y_val, y_pred)
                scores.append(metric_score)
            
        return scores


if __name__ == "__main__":
   #  data_root = "Data"
   #  processor = DataProcessor(data_root)
    
   # # Load and process data
   #  train_data, test_data = processor.load_data()
   #  print("Columns in the training data:", train_data.columns.tolist())
   #  print("Shape of training data:", train_data.shape)
    
   #  # Check for missing values
   #  missing_values = processor.check_missing_values(train_data)
   #  print(f"Missing values in training data: {missing_values}")
    
   #  # Clean data
   #  train_data = processor.clean_data(train_data)
   #  test_data = processor.clean_data(test_data)
   #  # Extract features and labels
   #  X_train, y_train = processor.extract_features_labels(train_data, is_train=True)
   #  X_test, y_test = processor.extract_features_labels(test_data, is_train=False)
    
   #  # Linear Regression Model
   #  linear_model = LinearRegression(learning_rate=0.16, max_iter=3000)
   #  loss_values_linear = linear_model.fit(X_train, y_train)
    
   #  # Logistic Regression Model
   #  logistic_model = LogisticRegression(learning_rate=0.1, max_iter=5000)
   #  y_train_binary = logistic_model.label_binarize(y_train)
   #  loss_values_logistic = logistic_model.fit(X_train, y_train_binary)
    
   #  # Cross Validation
   #  evaluator = ModelEvaluator(n_splits=5)
   #  linear_rmse_scores, _ = evaluator.cross_validation(linear_model, X_train, y_train)
   #  logistic_auroc_scores, logistic_f1_scores = evaluator.cross_validation(logistic_model, X_train, y_train)
    
   #  print(f'Linear Regression RMSE - Average: {np.mean(linear_rmse_scores):}, Std: {np.std(linear_rmse_scores):.2f}')
   #  print(f'Logistic Regression AUROC - Average: {np.mean(logistic_auroc_scores):}, Std: {np.std(logistic_auroc_scores):.2f}')
   #  print(f'Logistic Regression F1 Score - Average: {np.mean(logistic_f1_scores):}, Std: {np.std(logistic_f1_scores):.2f}')

    dp = DataProcessor("")
    train, test = dp.load_data()
    print(test.shape)
    train = dp.clean_data(train)
    test = dp.clean_data(test)
    lr = LinearRegression()
    lor = LogisticRegression()
    train_ft, train_target = dp.extract_features_labels(train)
    y_binary = lor.label_binarize(train_target)
    evaluator = ModelEvaluator()

    lin_scores = evaluator.cross_validation(lr, train_ft, train_target)
    log_scores = evaluator.cross_validation(lor, train_ft, train_target)
