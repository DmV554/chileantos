import datasets

# Define the paths to your datasets
paths = [
    "classification/Illegal.jsonl",
    "classification/Dark.jsonl",
    "classification/Gray.jsonl",
    "detection/Illegal.jsonl",
    "detection/Dark.jsonl",
    "detection/Gray.jsonl",
]

# Iterate over each dataset path
for path in paths:
    # Load the dataset
    dataset = datasets.load_dataset('json', data_files=path)
    print(dataset)

# Example code to train a model on the datasets
TRAIN_EXAMPLE = True


def train(paths):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import SVC
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.metrics import classification_report, f1_score
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.model_selection import ParameterSampler
    from scipy.stats import uniform
    import numpy as np

    # Iterate over each dataset path
    for path in paths:
        print(f"Training SVM classifier with TF-IDF features and custom hyperparameter search on {path}")
        # Load the dataset
        dataset = datasets.load_dataset('json', data_files=path)

        # Filter by split
        train_data = dataset['train'].filter(lambda x: x['split'] == 'train')
        validation_data = dataset['train'].filter(lambda x: x['split'] == 'validation')
        test_data = dataset['train'].filter(lambda x: x['split'] == 'test')

        # Extract text and labels
        train_texts = train_data['text']
        train_labels = train_data['labels']
        validation_texts = validation_data['text']
        validation_labels = validation_data['labels']
        test_texts = test_data['text']
        test_labels = test_data['labels']

        # Binarize the labels for multilabel classification
        mlb = MultiLabelBinarizer()
        train_labels_binarized = mlb.fit_transform(train_labels)
        validation_labels_binarized = mlb.transform(validation_labels)
        test_labels_binarized = mlb.transform(test_labels)

        # Preprocess text data using TF-IDF
        vectorizer = TfidfVectorizer(max_features=5000)
        # Fit on training + validation data to ensure consistent feature space
        X_train = vectorizer.fit_transform(train_texts + validation_texts)
        X_test = vectorizer.transform(test_texts)

        # Define the parameter distribution for hyperparameter search
        param_distributions = {
            'C': uniform(0.1, 10),  # Regularization parameter
            'kernel': ['linear', 'rbf'],  # Kernel type
            'gamma': ['scale', 'auto']  # Kernel coefficient for 'rbf'
        }

        # Setup the OneVsRestClassifier with SVC
        base_svm = SVC()
        svm = OneVsRestClassifier(base_svm)

        # Custom hyperparameter search with validation set integration
        n_iter_search = 10
        best_score = -np.inf
        best_params = None

        for params in ParameterSampler(param_distributions, n_iter=n_iter_search, random_state=42):
            # Set parameters
            svm.set_params(estimator__C=params['C'], estimator__kernel=params['kernel'], estimator__gamma=params['gamma'])

            # Fit on training data (excluding validation data here)
            svm.fit(X_train[:len(train_texts)], train_labels_binarized)

            # Evaluate on validation set
            validation_predictions = svm.predict(X_train[len(train_texts):])
            score = f1_score(validation_labels_binarized, validation_predictions, average='weighted')

            # Update best parameters if needed
            if score > best_score:
                best_score = score
                best_params = params

        print(f"Best parameters for {path}: {best_params} with validation F1 score: {best_score}")

        # Train the final model with best parameters on combined train+validation data
        svm.set_params(estimator__C=best_params['C'], estimator__kernel=best_params['kernel'],
                       estimator__gamma=best_params['gamma'])
        # Re-fit on combined train + validation data
        svm.fit(X_train, mlb.fit_transform(train_labels + validation_labels))

        # Evaluate on the test set
        test_predictions = svm.predict(X_test)
        target_names = [str(cls) for cls in mlb.classes_]
        print("Test classification report:")
        print(classification_report(test_labels_binarized, test_predictions, target_names=target_names))

if TRAIN_EXAMPLE:
    train(paths)