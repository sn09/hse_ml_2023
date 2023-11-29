import numpy as np
import pandas as pd


def _add_torque(df: pd.DataFrame) -> pd.DataFrame:
    import re
    kgm_coef = 9.80665

    torque_values = df["torque"].str.extractall(r"\D*([\d.,]+)\D*")
    torque_values["value"] = torque_values[0].str.replace(",", "").astype(float)
    torque_values = torque_values.drop(columns=[0]).reset_index()
    torque_values = torque_values.groupby("level_0").agg(
        torque=("value", "min"),
        max_torque_rpm=("value", "max")
    ).reset_index().rename(columns={"level_0": "index"})
    torque_values.loc[torque_values.max_torque_rpm < 1000, "max_torque_rpm"] = np.nan


    torque_type = df["torque"].str.extract(r".*(nm|kgm).*", re.IGNORECASE).reset_index()
    torque_type["torque_type"] = torque_type[0].str.lower()
    torque_type = torque_type.drop(columns=[0])

    torque = pd.merge(torque_type, torque_values, on="index", how="left")
    torque.loc[torque.torque_type == "kgm", "torque"] *=  kgm_coef
    return torque[["torque", "max_torque_rpm"]]

def _preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy(deep=True)
    for col in ["mileage", "engine", "max_power"]:
        df_new[col] = pd.to_numeric(
            df_new[col].str.split(pat=r"[^\d.,]", n=1).str[0],
            errors="coerce"
        )
    df_new[["torque", "max_torque_rpm"]] = _add_torque(df_new)
    return df_new

def _add_features(data, clip=False):
    df = data.copy()
    # Избавимся от выбросов
    if clip:
        for col in ["km_driven", "max_torque_rpm"]:
            q = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=q)
            
    df["seats"] = df["seats"].astype("category")
    
    # Добавим марку машины
    df["brand"] = df["name"].str.split(n=1).str[0]
    df = df.drop(columns="name")
    
    # Добавим количество км за год (кажется что 100000 км за 2 года и за 10 разные вещи)
    year_diff = df["year"].max() - df["year"] + 1
    df["km_year_ratio"] = df["km_driven"] / year_diff

    # Добавим отношения фичей про мощность к объему двигателя
    df["torque_ratio"] = (df["torque"] / df["engine"]).fillna(0).replace(np.inf, 0)
    df["max_torque_rpm_ratio"] = (df["max_torque_rpm"] / df["engine"]).fillna(0).replace(np.inf, 0)
    
    df["good_conditions_flg"] = (
        (df["seller_type"] == "Individual") & (df["owner"].isin(['First Owner', 'Second Owner']))
    ).astype("category")
    
    return df

def preprocessor_func(x):
    X_prep = _preprocess_df(x)
    X_prep = _add_features(X_prep, clip=True)
    
    return X_prep
