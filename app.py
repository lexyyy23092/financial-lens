from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.charts import (
    plot_actual_predicted,
    plot_correlation_heatmap,
    plot_distribution,
    plot_forecast,
    plot_missing_values,
    plot_prices,
    plot_time_series,
)
from src.correlation import correlation_matrix_from_prices, strongest_correlations
from src.data_loader import (
    dataframe_to_csv_bytes,
    get_numeric_columns,
    infer_date_columns,
    load_uploaded_file,
    prepare_time_series,
)
from src.eda import (
    categorical_summary,
    dataset_overview,
    duplicate_count,
    iqr_outlier_report,
    missing_values_report,
    numeric_summary,
)
from src.forecasting import run_forecast
from src.market_data import ASSET_UNIVERSE, DEFAULT_CORRELATION_ASSETS, download_price_matrix, download_single_asset
from src.utils import format_metric, infer_frequency_label


st.set_page_config(
    page_title="MarketLens Analytics Dashboard",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .metric-card {
        padding: 1rem;
        border-radius: 0.75rem;
        background: rgba(120, 120, 120, 0.08);
        border: 1px solid rgba(120, 120, 120, 0.18);
    }
    .small-note {font-size: 0.9rem; opacity: 0.8;}
    </style>
    """,
    unsafe_allow_html=True,
)


FREQ_OPTIONS = {
    "Daily": "D",
    "Weekly": "W",
    "Monthly": "MS",
}

MODEL_OPTIONS = ["Naive", "Moving Average", "ARIMA", "Prophet", "XGBoost"]


@st.cache_data(show_spinner=False)
def load_sample_dataset() -> pd.DataFrame:
    rng = pd.date_range("2021-01-01", periods=900, freq="D")
    np.random.seed(42)
    trend = np.linspace(100, 180, len(rng))
    seasonal = 8 * np.sin(np.arange(len(rng)) * 2 * np.pi / 30)
    noise = np.random.normal(0, 4, len(rng))
    sales = trend + seasonal + noise
    marketing_spend = 5000 + 600 * np.sin(np.arange(len(rng)) * 2 * np.pi / 90) + np.random.normal(0, 300, len(rng))
    df = pd.DataFrame(
        {
            "date": rng,
            "sales": sales.round(2),
            "marketing_spend": marketing_spend.round(2),
            "orders": np.maximum(1, (sales * 8 + np.random.normal(0, 30, len(rng))).round()).astype(int),
            "region": np.random.choice(["North", "South", "East", "West"], size=len(rng)),
        }
    )
    return df


def render_metric_row(df: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Duplicate rows", f"{duplicate_count(df):,}")


def get_active_uploaded_df() -> pd.DataFrame | None:
    return st.session_state.get("uploaded_df")


def choose_date_and_target(df: pd.DataFrame, key_prefix: str):
    date_cols = infer_date_columns(df)
    numeric_cols = get_numeric_columns(df)

    if not date_cols:
        st.warning("No date-like column was detected. Please upload/select a dataset with a date column for forecasting.")
        return None, None
    if not numeric_cols:
        st.warning("No numeric target column was detected.")
        return None, None

    c1, c2 = st.columns(2)
    with c1:
        date_col = st.selectbox("Date column", date_cols, key=f"{key_prefix}_date")
    with c2:
        target_col = st.selectbox("Target column", numeric_cols, key=f"{key_prefix}_target")
    return date_col, target_col


st.title("📈 MarketLens: EDA, Forecasting & Correlation Dashboard")
st.caption("Upload your own dataset, forecast time series, or analyze cross-asset relationships across major indices and commodities.")

with st.sidebar:
    st.header("Dashboard Controls")
    st.markdown(
        """
        **Modules**
        - Upload data and run EDA
        - Forecast uploaded or market data
        - Correlate indices and commodities
        """
    )
    st.info("Stock and index forecasts are educational only, not financial advice.")

home_tab, eda_tab, forecast_tab, corr_tab, about_tab = st.tabs(
    ["🏠 Home", "📊 Upload & EDA", "🔮 Forecasting", "🧮 Correlation Matrix", "ℹ️ Methodology"]
)

with home_tab:
    st.subheader("Project Overview")
    st.write(
        "MarketLens is a Streamlit web app for exploratory data analysis, time-series forecasting, "
        "and cross-market correlation analysis. It is designed for project demonstrations and can be "
        "extended into a more advanced analytics platform."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 1. Upload & EDA")
        st.write("Upload CSV/XLSX data and automatically inspect shape, missing values, column types, distributions, outliers, and correlations.")
    with c2:
        st.markdown("### 2. Forecasting")
        st.write("Forecast either uploaded data or predefined market assets using baseline, moving average, ARIMA, Prophet, or XGBoost.")
    with c3:
        st.markdown("### 3. Correlation")
        st.write("Download major indices and commodities, calculate returns, and visualize return correlations through a heatmap.")

    st.markdown("### Suggested demo flow")
    st.code(
        """1. Open Upload & EDA and load the sample dataset.
2. Select date = date and target = sales in Forecasting.
3. Run Prophet or XGBoost if installed; otherwise use Naive/Moving Average/ARIMA.
4. Open Correlation Matrix and compare NIFTY 50, SENSEX, S&P 500, NASDAQ, Dow Jones, Gold, and Silver.""",
        language="text",
    )

with eda_tab:
    st.subheader("Upload Dataset & Automated EDA")

    c1, c2 = st.columns([2, 1])
    with c1:
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
    with c2:
        use_sample = st.toggle("Use sample dataset", value=False)

    if use_sample:
        df = load_sample_dataset()
        st.session_state["uploaded_df"] = df
        st.success("Sample dataset loaded.")
    elif uploaded_file is not None:
        try:
            df = load_uploaded_file(uploaded_file)
            st.session_state["uploaded_df"] = df
            st.success(f"Loaded {uploaded_file.name}")
        except Exception as exc:
            st.error(str(exc))
            df = None
    else:
        df = get_active_uploaded_df()

    if df is None:
        st.info("Upload a dataset or enable the sample dataset to begin.")
    else:
        render_metric_row(df)
        st.markdown("### Data Preview")
        st.dataframe(df.head(50), use_container_width=True)

        st.markdown("### Column Overview")
        st.dataframe(dataset_overview(df), use_container_width=True)

        missing_df = missing_values_report(df)
        st.markdown("### Missing Values")
        st.dataframe(missing_df, use_container_width=True)
        missing_fig = plot_missing_values(missing_df)
        if missing_fig:
            st.plotly_chart(missing_fig, use_container_width=True)
        else:
            st.success("No missing values detected.")

        st.markdown("### Numeric Summary")
        num_summary = numeric_summary(df)
        if num_summary.empty:
            st.info("No numeric columns found.")
        else:
            st.dataframe(num_summary, use_container_width=True)

        st.markdown("### Categorical Summary")
        cat_summary = categorical_summary(df)
        if cat_summary.empty:
            st.info("No categorical columns found.")
        else:
            st.dataframe(cat_summary, use_container_width=True)

        numeric_cols = get_numeric_columns(df)
        if numeric_cols:
            st.markdown("### Distribution Explorer")
            selected_num_col = st.selectbox("Choose numeric column", numeric_cols, key="eda_distribution_col")
            st.plotly_chart(plot_distribution(df, selected_num_col), use_container_width=True)

            st.markdown("### Outlier Report")
            st.dataframe(iqr_outlier_report(df), use_container_width=True)

            if len(numeric_cols) >= 2:
                st.markdown("### Numeric Correlation Heatmap")
                corr = df[numeric_cols].corr(numeric_only=True)
                st.plotly_chart(plot_correlation_heatmap(corr, "Uploaded Dataset Correlation Matrix"), use_container_width=True)

        date_cols = infer_date_columns(df)
        if date_cols and numeric_cols:
            st.markdown("### Time-Series Preview")
            t1, t2 = st.columns(2)
            with t1:
                date_col = st.selectbox("Date column for preview", date_cols, key="eda_date_preview")
            with t2:
                value_col = st.selectbox("Value column for preview", numeric_cols, key="eda_value_preview")

            preview_df = df[[date_col, value_col]].copy()
            preview_df[date_col] = pd.to_datetime(preview_df[date_col], errors="coerce")
            preview_df[value_col] = pd.to_numeric(preview_df[value_col], errors="coerce")
            preview_df = preview_df.dropna().sort_values(date_col)
            if not preview_df.empty:
                st.plotly_chart(plot_time_series(preview_df, date_col, value_col, f"{value_col} over time"), use_container_width=True)

with forecast_tab:
    st.subheader("Time-Series Forecasting")
    source = st.radio("Choose data source", ["Uploaded dataset", "Predefined market asset"], horizontal=True)

    freq_label = st.selectbox("Forecast frequency", list(FREQ_OPTIONS.keys()), index=0)
    freq = FREQ_OPTIONS[freq_label]
    horizon = st.slider("Forecast horizon", min_value=7, max_value=365, value=30, step=1)
    test_size = st.slider("Test size", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
    model_name = st.selectbox("Model", MODEL_OPTIONS, index=1)

    moving_average_window = 7
    if model_name == "Moving Average":
        moving_average_window = st.slider("Moving average window", min_value=3, max_value=60, value=7)

    ts_df = None
    forecast_label = None

    if source == "Uploaded dataset":
        df = get_active_uploaded_df()
        if df is None:
            st.info("Upload a dataset in the Upload & EDA tab first, or switch to predefined market asset.")
        else:
            date_col, target_col = choose_date_and_target(df, "forecast_uploaded")
            aggregation = st.selectbox("Aggregation after resampling", ["mean", "sum", "median", "last"], index=0)
            if date_col and target_col:
                try:
                    ts_df = prepare_time_series(df, date_col, target_col, freq=freq, aggregation=aggregation)
                    forecast_label = f"Uploaded data: {target_col}"
                    st.plotly_chart(plot_time_series(ts_df, "ds", "y", f"Prepared series: {target_col} ({infer_frequency_label(freq)})"), use_container_width=True)
                    st.caption(f"Prepared observations: {len(ts_df):,}")
                except Exception as exc:
                    st.error(str(exc))
    else:
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            asset_name = st.selectbox("Market asset", list(ASSET_UNIVERSE.keys()), index=list(ASSET_UNIVERSE.keys()).index("NIFTY 50"))
        with c2:
            start_date = st.date_input("Start date", date.today() - timedelta(days=365 * 5))
        with c3:
            end_date = st.date_input("End date", date.today())

        symbol = ASSET_UNIVERSE[asset_name]
        try:
            market_df = download_single_asset(symbol, str(start_date), str(end_date))
            market_ts = market_df[["Date", "Price"]].rename(columns={"Date": "ds", "Price": "y"})
            market_ts["ds"] = pd.to_datetime(market_ts["ds"])
            ts_df = prepare_time_series(market_ts, "ds", "y", freq=freq, aggregation="last")
            forecast_label = f"{asset_name} ({symbol})"
            st.plotly_chart(plot_time_series(ts_df, "ds", "y", f"Prepared price series: {forecast_label}"), use_container_width=True)
            st.caption(f"Prepared observations: {len(ts_df):,}")
        except Exception as exc:
            st.error(str(exc))

    if ts_df is not None:
        if st.button("Run Forecast", type="primary"):
            try:
                with st.spinner("Training model and generating forecast..."):
                    result = run_forecast(
                        model_name=model_name,
                        ts_df=ts_df,
                        horizon=horizon,
                        freq=freq,
                        test_size=test_size,
                        moving_average_window=moving_average_window,
                    )

                st.success(f"Forecast complete: {result.model_name}")
                st.caption(result.notes)

                m1, m2, m3 = st.columns(3)
                m1.metric("MAE", format_metric(result.metrics.get("MAE")))
                m2.metric("RMSE", format_metric(result.metrics.get("RMSE")))
                m3.metric("MAPE", format_metric(result.metrics.get("MAPE")), help="Mean absolute percentage error; unavailable when actual values are zero.")

                st.markdown("### Test Set: Actual vs Predicted")
                st.plotly_chart(plot_actual_predicted(result.test_predictions, f"Actual vs Predicted - {forecast_label}"), use_container_width=True)
                st.dataframe(result.test_predictions.tail(50), use_container_width=True)

                st.markdown("### Future Forecast")
                st.plotly_chart(plot_forecast(ts_df, result.future_forecast, f"Future Forecast - {forecast_label}"), use_container_width=True)
                st.dataframe(result.future_forecast, use_container_width=True)

                st.download_button(
                    "Download forecast CSV",
                    data=dataframe_to_csv_bytes(result.future_forecast),
                    file_name="forecast_output.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(str(exc))
                st.info("Try a simpler model, a longer dataset, or a shorter forecast horizon.")

with corr_tab:
    st.subheader("Major Indices & Commodity Correlation Matrix")
    st.write("Correlations are calculated on percentage returns, not raw price levels, to avoid misleading trend-driven relationships.")

    selected_assets = st.multiselect(
        "Choose assets",
        options=list(ASSET_UNIVERSE.keys()),
        default=DEFAULT_CORRELATION_ASSETS,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        corr_start = st.date_input("Correlation start date", date.today() - timedelta(days=365 * 5), key="corr_start")
    with c2:
        corr_end = st.date_input("Correlation end date", date.today(), key="corr_end")
    with c3:
        return_frequency = st.selectbox("Return frequency", ["Daily", "Weekly", "Monthly"])

    if len(selected_assets) < 2:
        st.warning("Select at least two assets.")
    elif st.button("Build Correlation Matrix", type="primary"):
        asset_map = {name: ASSET_UNIVERSE[name] for name in selected_assets}
        try:
            with st.spinner("Downloading market data and calculating returns..."):
                prices = download_price_matrix(asset_map, str(corr_start), str(corr_end))
                corr, returns = correlation_matrix_from_prices(prices, return_frequency=return_frequency)

            errors = prices.attrs.get("download_errors", [])
            if errors:
                st.warning("Some assets could not be downloaded:\n" + "\n".join(errors))

            st.markdown("### Price History")
            st.plotly_chart(plot_prices(prices.ffill(), "Aligned Price History"), use_container_width=True)

            st.markdown("### Return Correlation Heatmap")
            st.plotly_chart(plot_correlation_heatmap(corr, f"{return_frequency} Return Correlation Matrix"), use_container_width=True)
            st.dataframe(corr.style.format("{:.3f}"), use_container_width=True)

            st.markdown("### Strongest Relationships")
            st.dataframe(strongest_correlations(corr, top_n=10).style.format({"Correlation": "{:.3f}"}), use_container_width=True)

            st.download_button(
                "Download correlation matrix CSV",
                data=dataframe_to_csv_bytes(corr.reset_index().rename(columns={"index": "Asset"})),
                file_name="correlation_matrix.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(str(exc))
            st.info("Try a wider date range, fewer assets, or a different return frequency.")

with about_tab:
    st.subheader("Methodology & Notes")

    st.markdown(
        """
        ### EDA methodology
        The EDA module checks data shape, missing values, duplicate rows, column types, summary statistics, categorical summaries, outliers, distributions, and numeric correlations.

        ### Forecasting methodology
        The forecasting module converts selected data into a clean two-column time series: `ds` for date and `y` for target value. It creates a train/test split and reports MAE, RMSE, and MAPE where possible.

        Available models:
        - **Naive**: uses the most recent value as the next prediction.
        - **Moving Average**: uses the recent rolling average.
        - **ARIMA**: classic statistical forecasting model.
        - **Prophet**: trend and seasonality model with forecast intervals.
        - **XGBoost**: supervised machine-learning model using lag and rolling features.

        ### Correlation methodology
        The correlation page downloads adjusted/close prices, aligns dates, converts prices to percentage returns, and then calculates Pearson correlations. Return correlations are more appropriate than price-level correlations for financial assets.

        ### Disclaimer
        This dashboard is for analytics education and project demonstration. It should not be used as investment advice.
        """
    )
