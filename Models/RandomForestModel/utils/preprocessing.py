import pandas as pd
def charger_et_preparer_donnees(path_csv):
    df = pd.read_csv(path_csv)
    df = df.drop(columns=["Unnamed: 0"],errors="ignore")
    X = df.drop(columns=["etiquette"])
    y = df["etiquette"]
    X = X.fillna(0)
    return X , y