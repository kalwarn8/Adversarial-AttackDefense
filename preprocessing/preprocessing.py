import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(train_path="data/adult.data", test_path="data/adult.test"):

    columns = [
        "age","workclass","fnlwgt","education","education-num",
        "marital-status","occupation","relationship","race","sex",
        "capital-gain","capital-loss","hours-per-week","native-country","income"
    ]

    # -------------------------
    # Load training data
    # -------------------------
    train_data = pd.read_csv(
        train_path,
        names=columns,
        na_values=" ?"
    )

    # -------------------------
    # Load testing data
    # adult.test has header row
    # -------------------------
    test_data = pd.read_csv(
        test_path,
        names=columns,
        skiprows=1,
        na_values=" ?"
    )

    # -------------------------
    # Remove missing values
    # -------------------------
    train_data = train_data.dropna()
    test_data = test_data.dropna()

    # -------------------------
    # Fix label formatting
    # adult.test labels end with "."
    # -------------------------
    train_data["income"] = train_data["income"].apply(lambda x: 1 if ">50K" in x else 0)
    test_data["income"] = test_data["income"].apply(lambda x: 1 if ">50K" in x else 0)

    # -------------------------
    # Split features / labels
    # -------------------------
    X_train = train_data.drop("income", axis=1)
    y_train = train_data["income"]

    X_test = test_data.drop("income", axis=1)
    y_test = test_data["income"]

    # -------------------------
    # One-hot encode categorical features
    # -------------------------
    X_all = pd.concat([X_train, X_test])
    X_all = pd.get_dummies(X_all)

    # split back
    X_train = X_all.iloc[:len(X_train)]
    X_test = X_all.iloc[len(X_train):]

    # -------------------------
    # Feature scaling
    # -------------------------
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------------------------
    # Print dataset statistics
    # -------------------------
    print("Dataset Statistics")
    print("------------------")
    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])
    print("Number of features:", X_train.shape[1])

    
    return X_train, X_test, y_train, y_test