import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.x13 import x13_arima_analysis
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.filters.hp_filter import hpfilter
import os

# Если есть X13 в папке bin
os.environ["X13PATH"] = "./bin"

CAPITAL_ELASTICITY = 0.3
LABOUR_ELASTICITY = 0.7

st.set_page_config(layout="wide", page_title="Output Gap Dashboard")

st.title("Оценка разрыва выпуска")

uploaded_file = st.file_uploader("Загрузите Excel-файл с данными", type=["xlsx"])

def x13_seasonal_adjustment(series, freq='Q'):
    """Сезонная корректировка через X13 с фоллбэком на STL"""
    series = pd.Series(series.values, index=pd.to_datetime(series.index))
    period_index = pd.period_range(start=series.index[0], periods=len(series), freq=freq)
    series.index = period_index
    try:
        res = x13_arima_analysis(series)
        return pd.Series(res.seasadj.values, index=series.index.to_timestamp())
    except Exception:
        st.warning("X13 недоступен, используется STL-декомпозиция.")
        res = STL(series.to_timestamp(), period=4).fit()
        return pd.Series(res.seasonal.values, index=series.index.to_timestamp())

def calc_output_gap(df_capital, df_population, df_other):
    """Расчёт разрыва выпуска"""
    # Датафреймы
    df_capital['date'] = pd.to_datetime(df_capital['date'])
    df_population['date'] = pd.to_datetime(df_population['date'])
    df_other['date'] = pd.to_datetime(df_other['date'])

    # Фактический выпуск
    output = df_other[df_other['indicator'] == "Приведённая отгрузка"].sort_values('date')
    output_sa = x13_seasonal_adjustment(output.set_index('date')['value'], freq='Q')

    # Инвестиции
    investments = df_other[df_other['indicator'] == "Приведённые инвестиции в основной капитал"].sort_values('date')
    investments_sa = x13_seasonal_adjustment(investments.set_index('date')['value'], freq='Q')

    # Занятость
    employment = df_other[df_other['indicator'] == "Численность работников"].sort_values('date')

    # Капитал (годовой) → квартальный через накопление
    capital = df_capital[df_capital['indicator'] == "Реальные основные фонды за вычетом учётного износа"].sort_values('date')
    capital_q = capital['value'].iloc[0]
    depreciation_rate = 0.05 / 4  # фикс 5% в год
    capital_inventory = []
    for invest in investments_sa:
        capital_q = capital_q * (1 - depreciation_rate) + invest
        capital_inventory.append(capital_q)

    # Потенциальная занятость
    population = df_population[df_population['indicator'] == "Численность населения в трудоспособном возрасте, всего"].sort_values('date')
    pop_q = np.repeat(population['value'].values, 4)[:len(employment)]
    emp_ratio = employment['value'].values / pop_q
    _, emp_cycle = hpfilter(emp_ratio, lamb=1600)
    potential_employment = (emp_ratio - emp_cycle) * pop_q * (1 - 0.046)

    # Потенциальный выпуск
    ln_output = np.log(output['value'].values)
    ln_capital = np.log(capital_inventory)
    ln_potential_employment = np.log(potential_employment)
    ln_tfp = ln_output - CAPITAL_ELASTICITY * ln_capital - LABOUR_ELASTICITY * ln_potential_employment
    ln_tfp_cycle, ln_tfp_trend = hpfilter(ln_tfp, lamb=1600)
    ln_potential_output = ln_tfp_trend + CAPITAL_ELASTICITY * ln_capital + LABOUR_ELASTICITY * ln_potential_employment
    potential_output = np.exp(ln_potential_output)
    output_gap = output_sa.values - potential_output

    df = pd.DataFrame({
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
    return df

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    df_capital = pd.read_excel(xls, sheet_name='Capital')
    df_population = pd.read_excel(xls, sheet_name='Population')
    df_other = pd.read_excel(xls, sheet_name='Other_data')

    df = calc_output_gap(df_capital, df_population, df_other)

    # Ключевые показатели
    df['year'] = pd.DatetimeIndex(df['date']).year
    last_year = df['year'].max()
    prev_year = last_year - 1
    df_last_year = df.groupby('year').last().reset_index()

    gdp_growth = round((df_last_year.loc[df_last_year['year']==last_year,'output_sa'].values[0] /
                        df_last_year.loc[df_last_year['year']==prev_year,'output_sa'].values[0] - 1)*100, 1)
    output_gap_abs = round(df['output_gap_inventory'].iloc[-1],0)
    output_gap_pct = round(df['output_gap_inventory'].iloc[-1] / df['potential_output_inv'].iloc[-1] * 100, 1)

    st.subheader("Ключевые показатели")
    st.table(pd.DataFrame({
        "Показатель": [
            f"Прирост выпуска (без сезонности), {last_year} к {prev_year}",
            "Последний разрыв выпуска",
            "Последний разрыв выпуска (доля от потенциала)"
        ],
        "Значение": [
            f"{gdp_growth} %",
            f"{output_gap_abs:,.0f}".replace(',', ' '),
            f"{output_gap_pct} %"
        ]
    }))

    # Графики
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15,
                        subplot_titles=("Динамика выпуска в ценах 2016 года", "Разрыв выпуска"))

    # 1. Выпуск
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['output'], name="Факт",
        mode='lines+markers', marker=dict(size=6, color="white", line=dict(width=1, color="#62C358")),
        line=dict(color="#62C358"),
        hovertemplate="%{x|%Y-Q%q}<br>Выпуск: %{y:,.0f}"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['output_sa'], name="Факт (без сезон.)",
        mode='lines+markers', marker=dict(size=6, color="white", line=dict(width=1, color="#085800")),
        line=dict(color="#085800"),
        hovertemplate="%{x|%Y-Q%q}<br>Без сезонности: %{y:,.0f}"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['date'], y=df['potential_output_inv'], name="Потенциальный выпуск",
        mode='lines+markers', marker=dict(size=6, color="white", line=dict(width=1, color="#A30008")),
        line=dict(color="#A30008"),
        hovertemplate="%{x|%Y-Q%q}<br>Потенциальный: %{y:,.0f}"
    ), row=1, col=1)

    # 2. Разрыв
    colors_gap = np.where(df['output_gap_inventory'] >= 0, "#085800", "#A30008")
    fig.add_trace(go.Bar(
        x=df['date'], y=df['output_gap_inventory'], name="Разрыв",
        marker_color=colors_gap,
        hovertemplate="%{x|%Y-Q%q}<br>Разрыв: %{y:,.0f}"
    ), row=2, col=1)

    fig.update_layout(height=700, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Загрузите Excel-файл, чтобы начать расчёты.")
