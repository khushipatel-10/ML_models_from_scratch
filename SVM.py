import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

'''
Problem: University Admission Classification using SVMs

Instructions:
1. Do not use any additional libraries. Your code will be tested in a pre-built environment with only 
   the library specified in question instruction available. Importing additional libraries will result in 
   compilation errors and you will lose marks.

2. Fill in the skeleton code precisely as provided. You may define additional 
   default arguments or helper functions if necessary, but ensure the input/output format matches.
'''
class DataLoader:
    '''
    Put your call to class methods in the __init__ method. Autograder will call your __init__ method only. 
    '''
    
    def __init__(self, data_path: str):
        """
        Initialize data processor with paths to train dataset. You need to have train and validation sets processed.
        
        Args:
            data_path: absolute path to your data file
        """
        self.train_data = pd.DataFrame()
        self.val_data = pd.DataFrame()
        
        # TODO：complete your dataloader here!

        df = pd.read_csv(data_path)
        df = self.create_binary_label(df)
        self.split_data(df)
    
    def create_binary_label(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Create a binary label for the training data.
        '''
        median_value = df['Chance of Admit'].median()
        df['label'] = (df['Chance of Admit'] > median_value).astype(int)
        return df
    
    def split_data(self, df: pd.DataFrame):
        train_size = int(len(df) * 0.8)
        self.train_data = df[:train_size]
        self.val_data = df[train_size:]

class SVMTrainer:
    def __init__(self):
        pass

    def train(self, X_train: np.ndarray, y_train: np.ndarray, kernel: str, **kwargs) -> SVC:
        '''
        Train the SVM model with the given kernel and parameters.

        Parameters:
            X_train: Training features
            y_train: Training labels
            kernel: Kernel type
            **kwargs: Additional arguments you may use
        Returns:
            SVC: Trained sklearn.svm.SVC model
        '''
        if kernel == 'poly':
            model = SVC(kernel=kernel, degree=3, **kwargs)
        else:
            model = SVC(kernel=kernel, **kwargs)
        model.fit(X_train, y_train)
        return model
    
    def get_support_vectors(self,model: SVC) -> np.ndarray:
        '''
        Get the support vectors from the trained SVM model.
        '''
        return model.support_vectors_

def plot_decision_boundary(model, X, y, feature_pair, kernel):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel(feature_pair[0])
    plt.ylabel(feature_pair[1])
    plt.title(f'Decision Boundary for {kernel} kernel with features {feature_pair}')
    plt.show()    

def main():
    global my_best_model

    data_loader = DataLoader('data-2.csv')
    train_data = data_loader.train_data
    val_data = data_loader.val_data

    features = [('CGPA', 'SOP'), ('CGPA', 'GRE Score'), ('SOP', 'LOR'), ('LOR', 'GRE Score')]
    kernels = ['linear', 'rbf', 'poly']
    trainer = SVMTrainer()

    best_accuracy = 0
    best_model = None
    best_feature_comb = None
    best_kernel = None

    for kernel in kernels:
        for feature_pair in features:
            X_train = train_data[list(feature_pair)].values
            y_train = train_data['label'].values
            X_val = val_data[list(feature_pair)].values
            y_val = val_data['label'].values

            if kernel == 'rbf':
                model = trainer.train(X_train, y_train, kernel=kernel, C=10, gamma=0.1)
            elif kernel == 'poly':
                model = trainer.train(X_train, y_train, kernel=kernel, C=5, gamma=0.2)
            else:
                model = trainer.train(X_train, y_train, kernel=kernel, C=1)

            # Identify support vectors
            support_vectors = trainer.get_support_vectors(model)
            print(f"Support Vectors for {kernel} kernel with features {feature_pair}:")
            print(support_vectors)

            # Visualize predictions on training set
            plot_decision_boundary(model, X_train, y_train, feature_pair, kernel)

            # Validate the model on the validation set
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            print(f"Accuracy for {kernel} kernel with features {feature_pair}: {accuracy}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_feature_comb = feature_pair
                best_kernel = kernel

    print(f"Best Model: {best_kernel} kernel with features {best_feature_comb} and accuracy: {best_accuracy}")

    my_best_model = best_model 

'''        
Initialize my_best_model with the best model you found.
'''

my_best_model = SVC(kernel='poly', degree=3, C=5, gamma=0.2)

if __name__ == "__main__":
    print("Hello, World!")
    main()
    my_best_model=SVC(kernel='poly', degree=3, C=5, gamma=0.2)