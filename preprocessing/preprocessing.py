import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(train_path="data/adult.data", test_path="data/adult.test"):

    columns = [
        "age","workclass","fnlwgt","education","education-num",
        "marital-status","occupation","relationship","race","sex",
        "capital-gain","capital-loss","hours-per-week","native-country","income"
    ]

    train_data = pd.read_csv(train_path, names=columns, na_values=" ?")
    test_data  = pd.read_csv(test_path,  names=columns, skiprows=1, na_values=" ?")

    train_data = train_data.dropna()
    test_data  = test_data.dropna()

    train_data["income"] = train_data["income"].apply(lambda x: 1 if ">50K" in x else 0)
    test_data["income"]  = test_data["income"].apply( lambda x: 1 if ">50K" in x else 0)

    X_train = train_data.drop("income", axis=1)
    y_train = train_data["income"]
    X_test  = test_data.drop("income", axis=1)
    y_test  = test_data["income"]

    # One-hot encode on combined to align columns
    X_all = pd.concat([X_train, X_test])
    X_all = pd.get_dummies(X_all)

    X_train = X_all.iloc[:len(X_train)]
    X_test  = X_all.iloc[len(X_train):]

    # Save feature names BEFORE scaling — needed by constraints + propagation
    feature_names = list(X_train.columns)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print("Dataset Statistics")
    print("------------------")
    print(f"Training samples : {X_train.shape[0]}")
    print(f"Testing samples  : {X_test.shape[0]}")
    print(f"Features         : {X_train.shape[1]}")

    # Returns 5 values now — feature_names is the new addition
    return X_train, X_test, y_train, y_test, feature_names
