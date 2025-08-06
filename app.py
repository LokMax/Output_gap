import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.filters.hp_filter import hpfilter
from datetime import datetime

CAPITAL_ELASTICITY = 0.3
LABOUR_ELASTICITY = 0.7

st.set_page_config("Output Gap Dashboard", layout="wide")

st.title("Оценка разрыва выпуска")

uploaded_file = st.file_uploader("Загрузите Excel-файл с данными (.xlsx)", type=["xlsx"])

@st.cache_data
def seasonal_adjust(series: pd.Series, freq='Q'):
    """Пробует X13, иначе использует STL."""
    try:
        from statsmodels.tsa.x13 import x13_arima_analysis
        result = x13_arima_analysis(series, freq='Q')
        return pd.Series(result.seasadj, index=series.index)
    except Exception:
        st.warning("X13 недоступен, используется STL-декомпозиция.")
        stl = STL(series, period=4 if freq == 'Q' else 12, robust=True)
        res = stl.fit()
        return pd.Series(series - res.seasonal, index=series.index)

def calc_output_gap(df_capital, df_population, df_other):
    df_capital['date'] = pd.to_datetime(df_capital['date'])
    df_population['date'] = pd.to_datetime(df_population['date'])
    df_other['date'] = pd.to_datetime(df_other['date'])

    # --- Выпуск ---
    output = df_other[df_other['indicator'] == "Приведённая отгрузка"].sort_values('date')
    output_series = pd.Series(output['value'].values, index=output['date'])
    output_sa = seasonal_adjust(output_series)

    # --- Инвестиции ---
    investments = df_other[df_other['indicator'] == "Приведённые инвестиции в основной капитал"].sort_values('date')
    invest_series = pd.Series(investments['value'].values, index=investments['date'])
    invest_sa = seasonal_adjust(invest_series)

    # --- Занятость ---
    employment = df_other[df_other['indicator'] == "Численность работников"].sort_values('date')

    # --- Капитал по инвентаризационному методу ---
    capital = df_capital[df_capital['indicator'] == "Реальные основные фонды за вычетом учётного износа"].sort_values('date')
    capital_q = capital['value'].iloc[0]
    depreciation_rate = 0.05 / 4

    capital_inventory = []
    for inv in invest_sa:
        capital_q = capital_q * (1 - depreciation_rate) + inv
        capital_inventory.append(capital_q)

    # --- Потенциальная занятость ---
    population = df_population[df_population['indicator'] == "Численность населения в трудоспособном возрасте, всего"].sort_values('date')
    pop_q = np.repeat(population['value'].values, 4)

    # Выравниваем длину
    if len(pop_q) < len(employment):
        pop_q = np.pad(pop_q, (0, len(employment)-len(pop_q)), mode='edge')
    else:
        pop_q = pop_q[:len(employment)]

    emp_ratio = employment['value'].values / pop_q
    _, emp_cycle = hpfilter(emp_ratio, lamb=1600)
    potential_employment = (emp_ratio - emp_cycle) * pop_q * (1 - 0.046)

    # --- Потенциальный выпуск ---
    ln_output = np.log(output['value'].values)
    ln_capital = np.log(capital_inventory)
    ln_potential_employment = np.log(potential_employment)

    ln_tfp = ln_output - CAPITAL_ELASTICITY*ln_capital - LABOUR_ELASTICITY*ln_potential_employment
    _, ln_tfp_trend = hpfilter(ln_tfp, lamb=1600)

    ln_potential_output = ln_tfp_trend + CAPITAL_ELASTICITY*ln_capital + LABOUR_ELASTICITY*ln_potential_employment
    potential_output = np.exp(ln_potential_output)
    output_gap = output_sa.values - potential_output

    return pd.DataFrame({
        'date': output['date'].values,
        'output': output['value'].values,
        'output_sa': output_sa.values,
        'investments': investments['value'].values,
        'capital_inventory': capital_inventory,
        'employment': employment['value'].values,
        'potential_employment': potential_employment,
        'potential_output_inv': potential_output,
        'output_gap_inventory': output_gap
    })

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    df_capital = pd.read_excel(xls, sheet_name="Capital")
    df_population = pd.read_excel(xls, sheet_name="Population")
    df_other = pd.read_excel(xls, sheet_name="Other_data")

    df = calc_output_gap(df_capital, df_population, df_other)

    # --- Графики ---
    colors_gap = np.where(df['output_gap_inventory'] >= 0, "#085800", "#A30008")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Динамика выпуска", "Разрыв выпуска"))

    fig.add_trace(
        go.Scatter(x=df['date'], y=df['output'], mode='lines+markers', name='Факт',
                   line=dict(color="#62C358"), marker=dict(color="white", line=dict(color="#62C358", width=1))),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['output_sa'], mode='lines+markers', name='Факт (без сезон.)',
                   line=dict(color="#085800"), marker=dict(color="white", line=dict(color="#085800", width=1))),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['potential_output_inv'], mode='lines+markers', name='Потенциальный выпуск',
                   line=dict(color="#A30008"), marker=dict(color="white", line=dict(color="#A30008", width=1))),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=df['date'], y=df['output_gap_inventory'], name='Разрыв', marker=dict(color=colors_gap)),
        row=2, col=1
    )

    fig.update_layout(height=700, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # --- Таблица ключевых показателей ---
    df['year'] = pd.to_datetime(df['date']).dt.year
    df_last_year = df.groupby('year').last().reset_index()
    last_year = df_last_year['year'].max()
    prev_year = last_year - 1

    gdp_growth = round((df_last_year.loc[df_last_year['year']==last_year, 'output_sa'].values[0] /
                        df_last_year.loc[df_last_year['year']==prev_year, 'output_sa'].values[0] - 1) * 100, 1)
    output_gap_abs = round(df['output_gap_inventory'].iloc[-1], 0)
    output_gap_pct = round(output_gap_abs / df['potential_output_inv'].iloc[-1] * 100, 1)
    capital_growth = round((df_last_year.loc[df_last_year['year']==last_year, 'capital_inventory'].values[0] /
                            df_last_year.loc[df_last_year['year']==prev_year, 'capital_inventory'].values[0] - 1) * 100, 1)
    emp_growth = round((df_last_year.loc[df_last_year['year']==last_year, 'employment'].values[0] /
                        df_last_year.loc[df_last_year['year']==prev_year, 'employment'].values[0] - 1) * 100, 1)

    st.subheader("Ключевые показатели")
    st.table(pd.DataFrame({
        "Показатель": [
            f"Прирост выпуска (без сезонности), {last_year} к {prev_year}",
            "Последний разрыв выпуска",
            "Последний разрыв выпуска (% от потенциала)",
            f"Прирост капитала, {last_year} к {prev_year}",
            f"Прирост занятости, {last_year} к {prev_year}"
        ],
        "Значение": [
            f"{gdp_growth} %",
            f"{output_gap_abs:,.0f}".replace(",", " "),
            f"{output_gap_pct} %",
            f"{capital_growth} %",
            f"{emp_growth} %"
        ]
    }))
