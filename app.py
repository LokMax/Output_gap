import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.x13 import x13_arima_analysis
import datetime as dt
import tempfile
import os

st.set_page_config(page_title="Оценка разрыва выпуска", layout="wide")

CAPITAL_ELASTICITY = 0.3
LABOUR_ELASTICITY = 0.7

st.title("Оценка разрыва выпуска")

uploaded_file = st.sidebar.file_uploader("Загрузите Excel-файл с данными", type=["xlsx"])

def x13_seasonal_adjustment(series, freq='Q'):
    """
    Возвращает сезонно скорректированный ряд через X13 ARIMA-SEATS.
    series: pandas.Series с индексом DatetimeIndex
    """
    # statsmodels требует, чтобы индекс был периодическим
    series = pd.Series(series.values, index=pd.period_range(start=series.index[0], periods=len(series), freq=freq))
    
    # X13 работает только с положительными значениями
    if (series <= 0).any():
        series = series + abs(series.min()) + 1
    
    try:
        res = x13_arima_analysis(series)
        return res.seasadj
    except Exception as e:
        st.warning(f"Не удалось выполнить X13: {e}")
        return series  # возврат без корректировки

if uploaded_file is not None:
    # Чтение данных
    cap_data = pd.read_excel(uploaded_file, sheet_name="Capital")
    pop_data = pd.read_excel(uploaded_file, sheet_name="Population")
    other_data = pd.read_excel(uploaded_file, sheet_name="Other_data")

    industries = other_data[['industry_code', 'industry']].drop_duplicates().sort_values('industry_code')
    industry = st.sidebar.selectbox("Выберите отрасль", industries['industry'])
    code = industries.loc[industries['industry'] == industry, 'industry_code'].values[0]

    # --- Фильтрация по отрасли ---
    cap_data = cap_data[cap_data['industry_code'] == code]
    other_data = other_data[other_data['industry_code'] == code]

    # --- Приведение дат ---
    for df in [cap_data, pop_data, other_data]:
        df['date'] = pd.to_datetime(df['date'])

    # --- Основные ряды ---
    output = other_data[other_data['indicator'] == "Приведённая отгрузка"].copy()
    investments = other_data[other_data['indicator'] == "Приведённые инвестиции в основной капитал"].copy()
    employment = other_data[other_data['indicator'] == "Численность работников"].copy()
    capital = cap_data[cap_data['indicator'] == "Реальные основные фонды за вычетом учётного износа"].copy()
    depreciation = cap_data[cap_data['indicator'] == "Ставка учётного износа"]['value'].values[0] / 4

    # --- Сезонная корректировка X13 ---
    output['output_sa'] = x13_seasonal_adjustment(output['value'], freq='Q')
    investments['investments_sa'] = x13_seasonal_adjustment(investments['value'], freq='Q')

    # --- Капитал по методу инвентаризации ---
    investment_sa = investments['investments_sa'].values
    capital_inventory = np.zeros(len(investment_sa))
    capital_inventory[0] = capital['value'].iloc[0]
    for t in range(1, len(investment_sa)):
        capital_inventory[t] = capital_inventory[t-1] * (1 - depreciation) + investment_sa[t]

    # --- Потенциальная занятость ---
    population = pop_data[pop_data['indicator'] == "Численность населения в трудоспособном возрасте, всего"]
    population_qtr = np.repeat(population['value'].values, 4)[:len(employment)]
    employment_ratio = employment['value'].values / population_qtr
    cycle, trend = hpfilter(employment_ratio, lamb=1600)
    potential_employment = trend * population_qtr * (1 - 0.046)

    # --- Потенциальный выпуск ---
    ln_output = np.log(output['value'].values)
    ln_capital_inventory = np.log(capital_inventory)
    ln_potential_employment = np.log(potential_employment)

    ln_tfp_inventory = ln_output - CAPITAL_ELASTICITY * ln_capital_inventory - LABOUR_ELASTICITY * ln_potential_employment
    cycle, ln_tfp_trend = hpfilter(ln_tfp_inventory, lamb=1600)
    ln_potential_output_inventory = ln_tfp_trend + CAPITAL_ELASTICITY * ln_capital_inventory + LABOUR_ELASTICITY * ln_potential_employment
    potential_output_inventory = np.exp(ln_potential_output_inventory)
    output_gap_inventory = output['output_sa'].values - potential_output_inventory

    df = pd.DataFrame({
        "date": output['date'],
        "output": output['value'],
        "output_sa": output['output_sa'],
        "investments": investments['value'],
        "investments_sa": investments['investments_sa'],
        "capital_inventory": capital_inventory,
        "employment": employment['value'],
        "potential_employment": potential_employment,
        "potential_output_inv": potential_output_inventory,
        "output_gap_inventory": output_gap_inventory
    })

    # --- Ключевые показатели ---
    df['year'] = df['date'].dt.year
    df_last_year = df.groupby('year').last().reset_index()
    last_year = df_last_year['year'].max()
    prev_year = last_year - 1

    gdp_growth = round((df_last_year.loc[df_last_year['year']==last_year,'output_sa'].values[0] /
                        df_last_year.loc[df_last_year['year']==prev_year,'output_sa'].values[0] - 1)*100, 1)
    output_gap_abs = int(df['output_gap_inventory'].iloc[-1])
    output_gap_pct = round(df['output_gap_inventory'].iloc[-1] / df['potential_output_inv'].iloc[-1]*100, 1)
    capital_growth = round((df_last_year.loc[df_last_year['year']==last_year,'capital_inventory'].values[0] /
                            df_last_year.loc[df_last_year['year']==prev_year,'capital_inventory'].values[0] - 1)*100, 1)
    emp_growth = round((df_last_year.loc[df_last_year['year']==last_year,'employment'].values[0] /
                        df_last_year.loc[df_last_year['year']==prev_year,'employment'].values[0] - 1)*100, 1)

    st.sidebar.markdown("### Ключевые показатели")
    st.sidebar.table(pd.DataFrame({
        "Показатель": [
            f"Прирост выпуска (без сезонности), {last_year} к {prev_year}",
            "Последний разрыв выпуска",
            "Последний разрыв выпуска (доля от потенциала)",
            f"Прирост капитала, {last_year} к {prev_year}",
            f"Прирост занятости, {last_year} к {prev_year}"
        ],
        "Значение": [
            f"{gdp_growth} %",
            f"{output_gap_abs:,}".replace(",", " "),
            f"{output_gap_pct} %",
            f"{capital_growth} %",
            f"{emp_growth} %"
        ]
    }))

    # --- Вкладка 1: Разрыв выпуска ---
    tab1, tab2, tab3 = st.tabs(["Разрыв выпуска", "Занятость", "Капитал"])

    with tab1:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        # Линии выпуска
        fig.add_trace(go.Scatter(x=df['date'], y=df['output'], mode='lines+markers',
                                 name='Факт', line=dict(color="#62C358"),
                                 marker=dict(color='white', line=dict(width=1,color="#62C358"))), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['output_sa'], mode='lines+markers',
                                 name='Факт (без сезон.)', line=dict(color="#085800"),
                                 marker=dict(color='white', line=dict(width=1,color="#085800"))), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['potential_output_inv'], mode='lines+markers',
                                 name='Потенциальный выпуск', line=dict(color="#A30008"),
                                 marker=dict(color='white', line=dict(width=1,color="#A30008"))), row=1, col=1)
        # Разрыв
        colors_gap = np.where(df['output_gap_inventory'] >= 0, "#085800", "#A30008")
        fig.add_trace(go.Bar(x=df['date'], y=df['output_gap_inventory'], name='Разрыв', marker=dict(color=colors_gap)), row=2, col=1)
        fig.update_layout(height=700, title="Динамика выпуска и разрыва")
        st.plotly_chart(fig, use_container_width=True)
