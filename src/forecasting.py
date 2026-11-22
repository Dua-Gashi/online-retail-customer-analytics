import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from pathlib import Path

FEATURES = [
    "lag_1", "lag_2", "lag_7", "lag_14", "avg_7", "avg_14", "day_of_week", 
    "month", "is_weekend", "year_end_break", "year_sin", "year_cos"
]

FIGURES_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"


def build_daily_series(sales, returns=None, drop_last_day=False):
    """Seria ditore: nje rresht per cdo dite kalendarike.

    revenue     = te ardhurat bruto te shitjeve te dites
    refunds     = vlera e anulimeve te dites (negative ose zero)
    net_revenue = revenue + refunds
    orders      = numri i faturave unike te shitjeve
    is_open     = True nese ka pasur te pakten nje fature shitjeje
    Ditet pa shitje marrin 0 ne te gjitha kolonat numerike.
    """
    parts = [
        sales.groupby("Date")["Revenue"].sum().rename("revenue"),
        sales.groupby("Date")["Invoice"].nunique().rename("orders"),
    ]
    if returns is not None:
        parts.append(returns.groupby("Date")["Revenue"].sum().rename("refunds"))
    daily = pd.concat(parts, axis=1).sort_index()
    if "refunds" not in daily.columns:
        daily["refunds"] = 0.0

    if drop_last_day:
        daily = daily.iloc[:-1]

    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D", name="Date")
    daily = daily.reindex(full_index)

    daily["is_open"] = daily["orders"].notna()
    cols = ["revenue", "orders", "refunds"]
    daily[cols] = daily[cols].fillna(0)
    daily["orders"] = daily["orders"].astype(int)
    daily["net_revenue"] = daily["revenue"] + daily["refunds"]
    return daily


def validate_daily(daily):
    """Kontrollon qe seria ditore eshte e plote dhe e sakte."""
    assert daily.index.is_unique and daily.index.is_monotonic_increasing, "Indeksi nuk eshte unik ose i renditur"
    steps = daily.index.to_series().diff().dropna()
    assert (steps == pd.Timedelta(days=1)).all(), "Ka dite qe mungojne ne kalendar"
    assert (daily["revenue"] >= 0).all(), "Ka te ardhura bruto negative"
    assert (daily["refunds"] <= 0).all(), "Anulimet duhet te jene negative ose zero"
    assert not daily.isna().any().any(), "Ka vlera qe mungojne"
    assert (daily.loc[~daily["is_open"], "revenue"] == 0).all(), "Dite pa fature me te ardhura bruto"
    print("Kontrollet e serise ditore: OK")
    print("Dite me te ardhura neto negative:", int((daily["net_revenue"] < 0).sum()))
    

def features_for_dates(data, target_col="net_revenue"):
    """Krijon vecorite (X) dhe objektivin (target) nga seria ditore
    
    Cdo veqori per diten D perdor vetem informacion para dites D (lag, mesatare)
    ose informacion qe dihet paraprakisht nga kalendari (dita e javes, muaji, pushimi)
    """
    
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Indeksi duhet te jete DatetimeIndex")
    
    s = data[target_col].astype(float)
    ix = data.index
    
    f = pd.DataFrame(index=ix)
    f["lag_1"] = s.shift(1)
    f["lag_2"] = s.shift(2)
    f["lag_7"] = s.shift(7)
    f["lag_14"] = s.shift(14)
    f["avg_7"] = s.shift(1).rolling(7, min_periods=7).mean()
    f["avg_14"] = s.shift(1).rolling(14, min_periods=14).mean()
    f["day_of_week"] = ix.dayofweek
    f["month"] = ix.month
    f["is_weekend"] = (ix.dayofweek >= 5).astype(int)
    f["year_end_break"] = (((ix.month == 12) & (ix.day >= 24)) | ((ix.month == 1) & (ix.day <= 3))).astype(int)
    f["year_sin"] = np.sin(2 * np.pi * ix.dayofyear / 365.25)
    f["year_cos"] = np.cos(2 * np.pi * ix.dayofyear / 365.25)
    f["target"] = s
    return f

def make_dataset(daily):
    """Kthen tabelen e gatshme per model: vecorit, target dhe is_open, pa rreshta me vlera qe mungojne"""
    
    f = features_for_dates(daily)
    f["is_open"] = daily["is_open"].astype(bool)
    return f.dropna(subset=FEATURES + ["target"]) 

def validate_features(f):
    """Kontrollon qe vecorit perdorin vetem te dhena nga e kaluara (pa rrjedhje te dhenash)"""
    
    t = f["target"].to_numpy()
    assert np.allclose(f["lag_1"].to_numpy()[1:], t[:-1]), "lag_1 nuk eshte vlera e dites se meparshme"
    assert np.allclose(f["lag_7"].to_numpy()[7:], t[:-7]), "lag_7 nuk eshte vlera e 7 diteve me pare"
    for i in (20, 100, 300):
        assert np.isclose(f["avg_7"].iloc[i], t[i - 7:i].mean()), "avg_7 perdor diten aktuale ose te ardhmen"    
    assert f["day_of_week"].between(0, 6).all(), "dita e javes jashte intervalit"
    print("Kontrollet e vecorive: OK")    



@dataclass
class TrainingResult:
    model: RandomForestRegressor
    test_predictions: pd.DataFrame
    metrics: pd.DataFrame
    train_size: int
    test_size: int
    importance: pd.DataFrame


def split_chronological(data, test_fraction=0.2):
    """Ndarje kronologjike: pjesa e pare per trajnim, pjesa e fundit per testim""" 
    
    split = int(len(data) * (1 - test_fraction)) 
    return data.iloc[:split], data.iloc[split:] 

def regression_metrics(y_true, y_pred):
    """MAE dhe RMSE per nje seri parashikimesh"""
    
    return{
        "MAE": float(mean_absolute_error(y_true,y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred)))
    }
    
def build_metrics_table(test_view, candidates=None):
    """Tabela e metrikave per modelin dhe baseline_t, per te gjitha ditet dhe vetem per ditet me shitje""" 
    if candidates is None:
        candidates = [
            ("Random Forest", "prediction"),
            ("Baseline: dje", "baseline_yesterday"),
            ("Baseline: java e kaluar", "baseline_week")
        ]
    open_view = test_view[test_view["is_open"]]
    rows = {}
    for name, col in candidates:
        all_days = regression_metrics(test_view["actual"], test_view[col])
        open_days = regression_metrics(open_view["actual"], open_view[col])
        rows[name] = {
            "MAE": all_days["MAE"],
            "RMSE": all_days["RMSE"],
            "MAE_dite_me_shitje": open_days["MAE"],
            "RMSE_dite_me_shitje": open_days["RMSE"]
        }
    return pd.DataFrame(rows).T

def train_and_test(daily, n_estimators=300, random_state=42):
    """Trajnon Random Forest mbi pjesen e pare te kohes dhe e teston mbi pjesen e fundit"""
    
    data = make_dataset(daily)
    if len(data) < 110:
        raise ValueError(f"Ka vetem {len(data)} dite per model. Duhet te jene te pakten 110")
    
    train, test = split_chronological(data)
    assert train.index.max() < test.index.min(), "Trajnimi dhe testimi mbivendosen ne kohe"
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=12,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(train[FEATURES], train["target"])
    pred = model.predict(test[FEATURES])
    
    test_view = pd.DataFrame(
        {
            "actual": test["target"],
            "prediction": pred,
            "baseline_yesterday": test["lag_1"],
            "baseline_week": test["lag_7"],
            "is_open": test["is_open"],
        },
        index = test.index
    )
    
    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": model.feature_importances_
        }
    ).sort_values("importance", ascending=False)
    
    return TrainingResult(
        model=model,
        test_predictions=test_view,
        metrics=build_metrics_table(test_view),
        train_size=len(train),
        test_size=len(test),
        importance=importance,
    )
    
def make_linear_model():
    """Linear Regression me one-hot per diten e javes (numri 0-6 nuk eshte shkalle lineare)"""
    
    prep = ColumnTransformer(
        [("dow", OneHotEncoder(handle_unknown="ignore"), ["day_of_week"])],
        remainder="passthrough"
    )
    return make_pipeline(prep, LinearRegression())

def get_models(random_state=42):
    """Fjalor me modelet qe krahasohen"""
    
    return{
        "Linear Regression": make_linear_model(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=random_state
        ),
    }
    
def compare_models(daily):
    """Trajnon te gjiotha modelet mbi te njejtin trajnim dhe i teston mbi te njejtin testim"""
    
    data = make_dataset(daily)
    train, test = split_chronological(data)
    
    test_view = pd.DataFrame(
        {
            "actual": test["target"],
            "baseline_yesterday": test["lag_1"],
            "baseline_week": test["lag_7"],
            "is_open": test["is_open"],
        },
        index=test.index
    )
    
    models = get_models()
    for name, model in models.items():
        model.fit(train[FEATURES], train["target"])
        test_view[name] = model.predict(test[FEATURES])
        
    candidates = [(name, name) for name in models]
    candidates += [("Baseline: dje", "baseline_yesterday"), ("Baseline: java e kaluar", "baseline_week")]
    return build_metrics_table(test_view, candidates), test_view

def cross_validate_models(daily, n_splits=5):
    """Vleresim me TimeSeriesSplit: cdo fold trajnon mbi te kaluaren dhe teston mbi bllokun pasues"""
    
    data = make_dataset(daily)
    splitter = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    
    for fold, (train_idx, test_idx) in enumerate(splitter.split(data), start=1):
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]
        open_test = test[test["is_open"]]
        info = {
            "fold": fold,
            "train_dite": len(train),
            "test_start": test.index.min().date(),
            "test_end": test.index.max().date()
        }
        
        for name, model in get_models().items():
            model.fit(train[FEATURES], train["target"])
            pred = pd.Series(model.predict(test[FEATURES]), index=test.index)
            rows.append({
                **info,
                "model": name,
                "MAE": mean_absolute_error(test["target"], pred),
                "MAE_dite_me_shitje": mean_absolute_error(open_test["target"], pred.loc[open_test.index]),         
            })
            
        for name, col in [("Baseline: dje", "lag_1"), ("Baseline: java e kaluar", "lag_7")]:
            rows.append({
                **info,
                "model": name,
                "MAE": mean_absolute_error(test["target"], test[col]),
                "MAE_dite_me_shitje": mean_absolute_error(open_test["target"], open_test[col]),
            })
            
    return pd.DataFrame(rows)

def make_random_forest(random_state=42, n_jobs=1):
    """Random Forest me te njejtat hiperparametra si ne krahsim"""
    
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    
def train_final_model(daily, random_state=42):
    """Trajnon Random Forest mbi te gjitha ditet e disponueshme (per parashikimin e se ardhmes)"""
    
    data = make_dataset(daily)
    model = make_random_forest(random_state)
    model.fit(data[FEATURES], data["target"])
    return model

def _predict_next(model, history, next_day):
    """Parashikon nje dite: ndertohen veqorite nga 20 ditet e fundit plus diten qe parashikohet"""
    
    window = history["net_revenue"].tail(20)
    placeholder = pd.Series([np.nan], index=pd.DatetimeIndex([next_day]))
    series = pd.concat([window, placeholder]).to_frame("net_revenue")
    x = features_for_dates(series).loc[[next_day], FEATURES]
    if x.isna().any().any():
        raise ValueError("Mungojne ditet e kaluara te nevojshme per parashikimin")
    return float(model.predict(x)[0])

def predict_future(daily, model, days=7):
    """Parashikim rekursiv: cdo dite e re perdor parashikimet e diteve te meparshme si 'te verteta'."""

    if not 1 <= days <= 14:
        raise ValueError("Parashikimi lejohet per 1 deri 14 dite")
    if daily["net_revenue"].tail(14).isna().any():
        raise ValueError("14 ditet e fundit duhet te jene te plota")
    
    history = daily[["net_revenue"]].copy()
    records = []
    for _ in range(days):
        next_day = history.index[-1] + pd.Timedelta(days=1)
        estimate = _predict_next(model, history, next_day)
        history.loc[next_day, "net_revenue"] = estimate
        records.append({
            "date": next_day,
            "day": next_day.day_name(),
            "forecast": round(estimate, 0)
        })
    return pd.DataFrame(records).set_index("date")

def backtest_horizons(daily, max_h=7, random_state=42):
    """Per cdo date origjine ne periudhen e testimit parashikon 1..max_h dite perpara
    (rekursivisht) dhe e krahason me te verteten. Modeli trajnohet vetem mbi trajnimin
    """
    data = make_dataset(daily)
    train, test = split_chronological(data)
    model = make_random_forest(random_state)
    model.fit(train[FEATURES], train["target"])
    
    net = daily["net_revenue"]
    is_open = daily["is_open"]
    last_date = daily.index.max()
    origins = pd.date_range(
        test.index.min() - pd.Timedelta(days=1),
        last_date - pd.Timedelta(days=max_h),
        freq="D"
    )
    
    rows = []
    for origin in origins:
        history = daily.loc[:origin, ["net_revenue"]].copy()
        for h in range(1, max_h + 1):
            day = origin + pd.Timedelta(days=h)
            estimate = _predict_next(model, history, day)
            history.loc[day, "net_revenue"] = estimate
            rows.append({
                "origin": origin,
                "h": h,
                "day": day,
                "actual": net.loc[day],
                "prediction": estimate,
                "baseline_week": net.loc[day - pd.Timedelta(days=7)],
                "is_open": bool(is_open.loc[day])
            })
    return pd.DataFrame(rows)

def horizon_summary(hz):
    """MEA sipas horizontit h (sa dite perpara), per modelin dhe baseline-in 'java e kaluar'""" 
    
    hz = hz.copy()
    hz["gabim_modeli"] = (hz["actual"] - hz["prediction"]).abs()
    hz["gabim_baseline"] = (hz["actual"] - hz["baseline_week"]).abs()
    cols = ["gabim_modeli", "gabim_baseline"]
    all_days = hz.groupby("h")[cols].mean()
    open_days = hz[hz["is_open"]].groupby("h")[cols].mean()
    return pd.DataFrame({
        "MAE_modeli": all_days["gabim_modeli"],
        "MAE_baseline_java": all_days["gabim_baseline"],
        "MAE_modeli_dite_me_shitje": open_days["gabim_modeli"],
        "MAE_baseline_dite_me_shitje": open_days["gabim_baseline"],
    })
    
def save_forecast_figure(daily, forecast, folder=FIGURES_DIR, last_days=60):
    """Ruan grafikun: te ardhurat reale te ditet e fundit dhe parashikimet"""
    
    import matplotlib.pyplot as plt
    
    folder.mkdir(parents=True, exist_ok=True)
    recent = daily["net_revenue"].tail(last_days)
    xs = [recent.index[-1], *forecast.index]
    ys = [recent.iloc[-1], *forecast["forecast"]]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(recent.index, recent.values, label="Reale (neto)", marker="o", markersize=3)
    ax.plot(xs, ys, label="Parashikim", marker="o", markersize=4, linestyle="--", color="tab:red")
    ax.set_title("te ardhurat neto ditore: 60 ditet e fundit dhe parashikimi")
    ax.set_xlabel("Data")
    ax.set_ylabel("Te ardhurat neto (£)")
    ax.legend()
    fig.tight_layout()
    path = folder / "forecast_next_7_days.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"U ruajt grafiku: {path}")