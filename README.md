from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_vitals = scaler.fit_transform(df_vitals)

df_scaled = pd.DataFrame(
    scaled_vitals,
    columns=df_vitals.columns
)

df_scaled.head()
df_scaled.info()
Here is the image content transcribed into text:
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)


