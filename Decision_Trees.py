import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_curve, auc
 
'''
General Instructions:

1. Do not use any additional libraries. Your code will be tested in a pre-built environment with only 
the library above available.

2. You are expected to fill in the skeleton code precisely as per provided. On top of skeleton code given,
you may write whatever deemed necessary to complete the assignment. For example, you may define additional 
default arguments, class parameters, or methods to help you complete the assignment.

3. Some initial steps or definition are given, aiming to help you get started. As long as you follow 
the argument and return type, you are free to change them as you see fit.

4. Your code should be free of compilation errors. Compilation errors will result in 0 marks.
'''


'''
Problem A-1: Data Preprocessing and EDA
'''
class DataLoader:
    '''
    This class will be used to load the data and perform initial data processing. Fill in functions.
    You are allowed to add any additional functions which you may need to check the data. This class 
    will be tested on the pre-built environment with only numpy and pandas available.
    '''

    def __init__(self, data_root: str, random_state: int):
        '''
        Initialize the DataLoader class with the data_root path.
        Load data as pd.DataFrame, store as needed and initialize other variables.
        All dataset should save as pd.DataFrame.
        '''
        self.random_state = random_state
        np.random.seed(self.random_state)

        self.data = pd.read_csv(data_root, delimiter=";")

        print("Initial data loaded:")
        print(self.data.head())

        self.data_train = None
        self.data_valid = None

        self.data_prep()
        self.data_split()

    def data_split(self) -> None:
        '''
        You are asked to split the training data into train/valid datasets on the ratio of 80/20.
        Maintain class balance during the split.
        '''
        class_0 = self.data[self.data['y'] == 0]
        class_1 = self.data[self.data['y'] == 1]
        class_0 = class_0.sample(frac=1, random_state=self.random_state)
        class_1 = class_1.sample(frac=1, random_state=self.random_state)

        # Calculate the split index for 80/20
        split_0 = int(0.8 * len(class_0))
        split_1 = int(0.8 * len(class_1))

        # Split the classes into train and valid
        train_class_0 = class_0[:split_0]
        valid_class_0 = class_0[split_0:]
        train_class_1 = class_1[:split_1]
        valid_class_1 = class_1[split_1:]

        train_data = pd.concat([train_class_0, train_class_1]).reset_index(drop=True)
        valid_data = pd.concat([valid_class_0, valid_class_1]).reset_index(drop=True)

        self.data_train = train_data
        self.data_valid = valid_data

        print("Data split into train and valid:")
        print(f"Train shape: {self.data_train.shape}, Valid shape: {self.data_valid.shape}")

    def data_prep(self) -> None:
        '''
        You are asked to drop any rows with missing values and map categorical variables to numeric values. 
        '''
        print(f"Shape before cleaning: {self.data.shape}")
        self.data = self.data.dropna()
        select_categorical_columns = self.data.select_dtypes(include=['object']).columns
        
        for col in select_categorical_columns:
            self.data[col] = pd.factorize(self.data[col])[0]
        print(f"Shape after cleaning: {self.data.shape}")

    def extract_features_and_label(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        '''
        This function will be called multiple times to extract features and labels from train/valid/test 
        data.
        
        Expected return:
            X_data: np.ndarray of shape (n_samples, n_features) - Extracted features
            y_data: np.ndarray of shape (n_samples,) - Extracted labels
        '''
        X_data = data.drop(columns='y').values
        y_data = data['y'].values
        return X_data, y_data

    def plot_histograms(self):
        '''
        Plot histograms of all variables and provide a brief discussion.
        '''
        self.data_train.hist(bins=50, figsize=(20,15))
        plt.show()


'''
Problem A-2: Classification Tree Implementation
'''
class ClassificationTree:
    '''
    You are asked to implement a simple classification tree from scratch. This class will be tested on the
    pre-built environment with only numpy and pandas available.

    You may add more variables and functions to this class as you see fit.
    '''
    class Node:
        '''
        A data structure to represent a node in the tree.
        '''
        def __init__(self, split=None, left=None, right=None, prediction=None):
            '''
            split: tuple - (feature_idx, split_value, is_categorical)
                - For numerical features: split_value is the threshold
                - For categorical features: split_value is a set of categories for the left branch
            left: Node - Left child node
            right: Node - Right child node
            prediction: (any) - Prediction value if the node is a leaf
            '''
            self.split = split
            self.left = left
            self.right = right
            self.prediction = prediction 

        def is_leaf(self):
            return self.prediction is not None

    def __init__(self, random_state: int, max_depth: int = 5):
        self.random_state = random_state
        self.max_depth = max_depth
        np.random.seed(self.random_state)

        self.tree_root = None

    def split_crit(self, y: np.ndarray) -> float:
        '''
        Implement the impurity measure of your choice here. Return the impurity value.
        '''
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return 1 - np.sum(p ** 2)
        
    def build_tree(self, X: np.ndarray, y: np.ndarray) -> None:
        '''
        Implement the tree building algorithm here. You can recursively call this function to build the 
        tree. After building the tree, store the root node in self.tree_root.
        '''
        self.tree_root = self._build_tree(X, y)
        return self.tree_root 
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        return self.build_tree(X,y)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth=0):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return ClassificationTree.Node(prediction=np.bincount(y).argmax())

        best_split = self.search_best_split(X, y)
        if best_split is None:
            return ClassificationTree.Node(prediction=np.bincount(y).argmax())

        feature_idx, split_value, is_categorical = best_split

        if is_categorical:
            left_mask = X[:, feature_idx] == split_value
        else:
            left_mask = X[:, feature_idx] <= split_value

        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return ClassificationTree.Node(split=(feature_idx, split_value, is_categorical), left=left_child, right=right_child)

    def search_best_split(self, X: np.ndarray, y: np.ndarray):
        '''
        Implement the search for best split here.

        Expected return:
        - tuple(int, float): Best feature index and split value
        - None: If no split is found
        '''
        m, n = X.shape
        best_split = None
        best_crit = float('inf')

        for feature_idx in range(n):
            unique_values = np.unique(X[:, feature_idx])
            for split_value in unique_values:
                left_mask = X[:, feature_idx] <= split_value
                right_mask = ~left_mask

                if len(np.unique(left_mask)) == 1 or len(np.unique(right_mask)) == 1:
                    continue

                left_crit = self.split_crit(y[left_mask])
                right_crit = self.split_crit(y[right_mask])
                crit = (left_crit * len(y[left_mask]) + right_crit * len(y[right_mask])) / len(y)

                if crit < best_crit:
                    best_crit = crit
                    best_split = (feature_idx, split_value, False)

        return best_split

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Predict classes for multiple samples.
        
        Args:
            X: numpy array with the same columns as the training data
            
        Returns:
            np.ndarray: Array of predictions
        '''
        return np.array([self._predict_single(x, self.tree_root) for x in X])

    def _predict_single(self, x, node: Node):
        if node.is_leaf():
            return node.prediction

        feature_idx, split_value, is_categorical = node.split

        if is_categorical:
            if x[feature_idx] == split_value:
                return self._predict_single(x, node.left)
            else:
                return self._predict_single(x, node.right)
        else:
            if x[feature_idx] <= split_value:
                return self._predict_single(x, node.left)
            else:
                return self._predict_single(x, node.right)


def train_XGBoost() -> dict:
    '''
    See instruction for implementation details. This function will be tested on the pre-built environment
    with numpy, pandas, xgboost available.
    '''
    data_loader = DataLoader(data_root='bank.csv', random_state=42)

    X_train, y_train = data_loader.extract_features_and_label(data_loader.data_train)
    X_valid, y_valid = data_loader.extract_features_and_label(data_loader.data_valid)

    alpha_vals = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    best_alpha, best_f1 = train_xgboost_with_bootstrap(X_train, y_train, X_valid, y_valid, alpha_vals)

    print(f"Best Alpha: {best_alpha}, Best F1 Score: {best_f1}")
    scale_pos_weight = np.bincount(y_train)[0] / np.bincount(y_train)[1]  # Compute class imbalance ratio

    # Train the model with the best alpha found through bootstrapping
    my_best1_model = XGBClassifier(reg_alpha=best_alpha, max_depth=12, learning_rate=0.06, subsample=0.6, n_estimators=900, scale_pos_weight=scale_pos_weight, random_state=22)
    my_best1_model.fit(X_train, y_train)

    # Evaluate the model on the validation set
    y_pred_prob = my_best1_model.predict_proba(X_valid)[:, 1]
    fpr, tpr, _ = roc_curve(y_valid, y_pred_prob)
    auc_score = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_score:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

    print(f"AUC Score: {auc_score}")


def train_xgboost_with_bootstrap(X_train, y_train, X_valid, y_valid, alpha_vals, num_iterations=100):
    best_alpha = None
    best_f1 = 0

    for alpha in alpha_vals:
        f1_scores = []

        for _ in range(num_iterations):
            # Bootstrapping: create a bootstrapped sample from the training data
            bootstrap_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
            X_train_bootstrap = X_train[bootstrap_idx]
            y_train_bootstrap = y_train[bootstrap_idx]

            # Train the model on the bootstrapped sample
            model = XGBClassifier(reg_alpha=alpha, max_depth=12, learning_rate=0.06, subsample=0.6, n_estimators=900, random_state=22)
            model.fit(X_train_bootstrap, y_train_bootstrap)

            # Predict and evaluate the model
            y_pred = model.predict(X_valid)
            f1 = f1_score(y_valid, y_pred)
            f1_scores.append(f1)

        # Calculate the mean F1 score over all bootstrapping iterations
        mean_f1 = np.mean(f1_scores)
        print(f"Alpha: {alpha}, F1 Score: {mean_f1}")
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_alpha = alpha

    return best_alpha, best_f1
 
'''
Initialize the following variable with the best model you have found. This model will be used in testing 
in our pre-built environment.
'''
my_best_model = XGBClassifier(reg_alpha=1, max_depth=3, learning_rate=0.28, n_estimators=1000, random_state=8)

if __name__ == "__main__":
    print("Hello World!")
    train_XGBoost()
