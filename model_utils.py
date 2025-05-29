import os
import uuid
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
import re

# Location mapping based on provided data
LOCATION_MAPPING = {
    (26.473806, 80.348389): "Naveen Market",
    (26.434194, 80.322500): "Saket Nagar",
    (26.438889, 80.293000): "Mapple Kids School",
    (26.439306, 80.277500): "Kamadgiri Lawn",
    (26.435139, 80.336722): "M-Block, Kidwai Nagar",
    (26.495889, 80.288500): "KDA Market",
    (26.469806, 80.303389): "KMC Hospital",
    (26.454306, 80.364111): "M G Park",
    (26.458806, 80.233806): "New Transport Nagar",
    (26.459889, 80.260889): "KPR, Kanpur",
    (26.432389, 80.398694): "Jajmau",
    (26.453889, 80.290694): "Dada Nagar",
    (26.460389, 80.326611): "Jarib Chawki",
    (26.430000, 80.364000): "DMSRDE, Kanpur"
}

def dms_to_decimal(dms: str) -> float:
    """Convert DMS (degrees, minutes, seconds) string to decimal degrees."""
    try:
        dms = dms.strip()
        # Handle formats like "26°28'25.7\" N" or "80°17'34.8\"E"
        match = re.match(r'(\d+)°\s*(\d+)\'\s*(\d*\.?\d*)\"?\s*([NSEW])', dms)
        if not match:
            raise ValueError(f"Invalid DMS format: {dms}")
        
        degrees, minutes, seconds, direction = match.groups()
        degrees = float(degrees)
        minutes = float(minutes)
        seconds = float(seconds) if seconds else 0.0
        
        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
        if direction in ('S', 'W'):
            decimal = -decimal
        return round(decimal, 6)
    except Exception as e:
        raise ValueError(f"Failed to convert DMS '{dms}': {str(e)}")

def convert_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Latitude and Longitude columns from DMS to decimal degrees."""
    df = df.copy()
    lat_col = next((col for col in df.columns if col.strip().lower() in ['latitude', ' latitude']), None)
    lon_col = next((col for col in df.columns if col.strip().lower() in ['longitude', ' longitude']), None)
    
    if lat_col and lon_col:
        df[lat_col] = df[lat_col].apply(lambda x: dms_to_decimal(str(x)) if isinstance(x, str) else x)
        df[lon_col] = df[lon_col].apply(lambda x: dms_to_decimal(str(x)) if isinstance(x, str) else x)
        # Rename to standard names
        df.rename(columns={lat_col: 'Latitude', lon_col: 'Longitude'}, inplace=True)
    return df

def get_location_name(lat: float, lon: float) -> str:
    """Find the closest location name based on latitude and longitude."""
    lat, lon = round(lat, 6), round(lon, 6)
    for (loc_lat, loc_lon), name in LOCATION_MAPPING.items():
        if abs(loc_lat - lat) < 0.001 and abs(loc_lon - lon) < 0.001:
            return name
    return "Unknown"

def process_file(filepath: str, plot_dir: str) -> Dict[str, Any]:
    # Load and preprocess data
    data = pd.read_excel(filepath)
    data = convert_coordinates(data)
    drop_cols = ["L.No.", "Latitude", "Longitude", "Atm Pressure (mBar)", "Median (m)"]
    data = data.drop(columns=drop_cols, errors='ignore')
    TARGET = "Leq"
    X = data.drop(columns=[TARGET], errors='ignore')
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=X['Peak type'], random_state=42
    )

    # One-hot encode categorical variables
    cat_cols = ['Peak type', 'Landuse']
    enc = OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse=False)
    enc.fit(X_train[cat_cols])
    def transform(df):
        arr = enc.transform(df[cat_cols])
        return pd.concat([
            df.drop(columns=cat_cols),
            pd.DataFrame(arr, columns=enc.get_feature_names_out(cat_cols), index=df.index)
        ], axis=1)
    X_train2, X_test2 = transform(X_train), transform(X_test)

    # Remove low-variance features
    low_var = X_train2.var()[lambda v: v < 0.05].index
    X_train2.drop(columns=low_var, inplace=True)
    X_test2.drop(columns=low_var, inplace=True)

    # Scale numeric features
    num_cols = X_train2.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler().fit(X_train2[num_cols])
    X_train2[num_cols] = scaler.transform(X_train2[num_cols])
    X_test2[num_cols] = scaler.transform(X_test2[num_cols])

    # Define models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=500, min_samples_split=3, min_samples_leaf=2,
                                              max_features=0.7, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=700, learning_rate=0.05, max_depth=5,
                                                      subsample=0.8, min_samples_split=2, max_features=0.9,
                                                      random_state=42),
        'XGBoost': XGBRegressor(n_estimators=700, learning_rate=0.05, max_depth=6, subsample=0.8,
                               colsample_bytree=0.8, gamma=0.1, reg_alpha=0.1, reg_lambda=5,
                               random_state=42, n_jobs=-1)
    }

    output = {}
    for name, model in models.items():
        model.fit(X_train2, y_train)
        preds = model.predict(X_test2)
        metrics = {'Overall': {
            'R2': round(r2_score(y_test, preds), 4),
            'MSE': round(mean_squared_error(y_test, preds), 4),
            'MAE': round(mean_absolute_error(y_test, preds), 4)
        }}
        os.makedirs(plot_dir, exist_ok=True)
        img_list = []
        def save_plot(y_true, y_pred, tag):
            fname = f"{name.replace(' ', '_')}_{tag}.png"
            path = os.path.join(plot_dir, fname)
            plt.figure(figsize=(5,5))
            sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
            plt.xlabel('Obs Leq'); plt.ylabel('Pred Leq'); plt.title(f"{name} - {tag}")
            plt.tight_layout(); plt.savefig(path); plt.close()
            img_list.append(fname)

        save_plot(y_test, preds, 'Overall')
        for pk in X_test['Peak type'].unique():
            mask = X_test['Peak type'] == pk
            pr = model.predict(X_test2[mask])
            metrics[pk] = {
                'R2': round(r2_score(y_test[mask], pr), 4),
                'MSE': round(mean_squared_error(y_test[mask], pr), 4),
                'MAE': round(mean_absolute_error(y_test[mask], pr), 4)
            }
            save_plot(y_test[mask], pr, str(pk))

        output[name] = {'metrics': metrics, 'plots': img_list}
    return output

def process_predict_file(filepath: str, csv_folder: str) -> Tuple[pd.DataFrame, str]:
    # Define drop columns to match main code
    drop_cols = ["L.No.", "S.No.", "S. No.", "Latitude", "Longitude", "Atm Pressure (mBar)", "Median (m)", "Wind Speed (m/s)", "Relative Humidity (%)", "Temperature (0C)"]
    TARGET = "Leq"

    # Load and preprocess training data
    train_data = pd.read_excel("Dataset_Noise_cleaned_synthetic3.xlsx")
    train_data = convert_coordinates(train_data)
    train_data = train_data.drop(columns=drop_cols, errors='ignore')
    X_full = train_data.drop(columns=[TARGET], errors='ignore')
    y_full = train_data[TARGET]
    X_train, _, y_train, _ = train_test_split(
        X_full, y_full, test_size=0.2, stratify=X_full['Peak type'], random_state=42
    )

    # Encode categorical columns
    categorical_cols = ["Peak type", "Landuse"]
    encoder = OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse=False)
    encoder.fit(X_train[categorical_cols])

    def prepare(df):
        arr = encoder.transform(df[categorical_cols])
        return pd.concat([
            df.drop(columns=categorical_cols),
            pd.DataFrame(arr, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)
        ], axis=1)

    X_train_encoded = prepare(X_train)

    # Drop low-variance features based on X_train
    variance = X_train_encoded.var()
    low_variance_cols = variance[variance < 0.05].index.tolist()
    X_train_preprocessed = X_train_encoded.drop(columns=low_variance_cols, errors='ignore')

    # Scale numerical features
    numeric_cols = X_train_preprocessed.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler().fit(X_train_preprocessed[numeric_cols])
    X_train_scaled = X_train_preprocessed.copy()
    X_train_scaled[numeric_cols] = scaler.transform(X_train_preprocessed[numeric_cols])

    # Train RandomForest model
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(rf,
        {"n_estimators":[500], "min_samples_split":[3], "min_samples_leaf":[2],
         "max_features":[0.7], "bootstrap":[True]},
        n_iter=40, cv=5, scoring="neg_mean_squared_error", random_state=42)
    search.fit(X_train_scaled, y_train)
    best = search.best_estimator_

    # Load and preprocess new data
    raw = pd.read_excel(filepath)
    raw = convert_coordinates(raw)
    X_data2_full = raw.drop(columns=drop_cols, errors='ignore')
    X_data2_encoded = prepare(X_data2_full)
    X_data2_preprocessed = X_data2_encoded.drop(columns=low_variance_cols, errors='ignore')

    # Align columns with X_train_preprocessed
    train_cols = X_train_preprocessed.columns
    missing_cols = set(train_cols) - set(X_data2_preprocessed.columns)
    for col in missing_cols:
        X_data2_preprocessed[col] = 0
    X_data2_preprocessed = X_data2_preprocessed[train_cols]

    # Scale
    X_data2_scaled = X_data2_preprocessed.copy()
    X_data2_scaled[numeric_cols] = scaler.transform(X_data2_preprocessed[numeric_cols])

    # Handle infinite values and NaNs
    X_data2_scaled.replace([np.inf, -np.inf], 0, inplace=True)
    X_data2_scaled.fillna(0, inplace=True)

    # Predict
    preds = best.predict(X_data2_scaled)

    # Create output dataframe
    longitude_col = "Longitude"
    latitude_col = "Latitude"
    # Search for serial number column with broader matching
    possible_lno_cols = ["L.No.", "S.No.", "S. No.", "Serial No.", "Serial Number", "ID", "Index"]
    lno_col = next((col for col in raw.columns if col.strip().lower() in [c.lower() for c in possible_lno_cols]), None)
    
    # Required columns for output
    required_cols = [latitude_col, longitude_col, "Peak type", "Landuse"]
    if lno_col:
        required_cols.insert(0, lno_col)
    
    # Validate columns
    missing_cols = [col for col in required_cols if col not in raw.columns]
    if missing_cols:
        raise KeyError(f"Required columns missing in input file: {missing_cols}")

    df_out = raw[required_cols].copy()
    df_out.rename(columns={lno_col: "L.No."} if lno_col else {}, inplace=True)
    df_out["Noise_value"] = np.round(preds, 3)

    # Add Location column
    df_out["Location"] = df_out.apply(lambda row: get_location_name(row["Latitude"], row["Longitude"]), axis=1)

    # If no L.No. column was found, generate one
    if not lno_col:
        df_out.insert(0, "L.No.", [f"L{i}" for i in range(1, len(df_out) + 1)])

    # Group and average
    grouped = df_out.groupby(
        ["Latitude", "Longitude", "Peak type", "Landuse", "Location"],
        as_index=False
    )["Noise_value"].mean().round(3)

    # Add sequential L.No.
    grouped.insert(0, "L.No.", [f"L{i}" for i in range(1, len(grouped) + 1)])

    # Noise limit check
    def check(row):
        v, z = row["Noise_value"], row["Landuse"].strip().lower()
        if z == "residential":  return "Under the Limit" if v < 55 else "Over the Limit"
        if z == "commercial":   return "Under the Limit" if v < 65 else "Over the Limit"
        if z == "industrial":   return "Under the Limit" if v < 75 else "Over the Limit"
        if z in ("silence", "silence zone"): return "Under the Limit" if v < 50 else "Over the Limit"
        return "Unknown"
    grouped["Noise Limit Check"] = grouped.apply(check, axis=1)

    # Final CSV
    final = grouped[[
        "L.No.", "Location", "Latitude", "Longitude", "Peak type", "Landuse", "Noise_value", "Noise Limit Check"
    ]]
    csv_name = f"{uuid.uuid4()}_predictions.csv"
    final.to_csv(os.path.join(csv_folder, csv_name), index=False)
    return final, csv_name