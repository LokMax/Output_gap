import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# === НАСТРОЙКА СТРАНИЦЫ ===
st.set_page_config(page_title="Анализ выпуска", layout="wide")

st.title("📊 Панель анализа выпуска и разрыва")

# === ЗАГРУЗКА ФАЙЛА ===
uploaded_file = st.file_uploader("Загрузите Excel/CSV с данными", type=["xlsx", "csv"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, parse_dates=["date"])
    else:
        df = pd.read_excel(uploaded_file, parse_dates=["date"])
    
    df = df.sort_values("date")

    # --- Вычисление вспомогательных колонок ---
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["quarter_label"] = df["year"].astype(str) + " Q" + df["quarter"].astype(str)
    
    # Пример прироста ВВП
    annual_gdp = df.groupby("year")["output_sa"].mean()
    if len(annual_gdp) >= 2:
        gdp_growth = round((annual_gdp.iloc[-1] / annual_gdp.iloc[-2] - 1) * 100, 1)
    else:
        gdp_growth = None

    # === Блок с показателями ===
    st.sidebar.subheader("📈 Ключевые показатели")
    if gdp_growth is not None:
        st.sidebar.metric("Прирост ВВП (без сезон.)", f"{gdp_growth}%")
    st.sidebar.metric("Последняя дата", df["date"].max().strftime("%Y-%m-%d"))
    st.sidebar.metric("Фактический выпуск", f"{df['output_sa'].iloc[-1]:,.0f}".replace(",", " "))

    # === Вкладки ===
    tab1, tab2, tab3 = st.tabs(["Разрыв выпуска", "Занятость", "Капитал и инвестиции"])

    # === Вкладка 1: Выпуск + Разрыв ===
    with tab1:
        st.subheader("Динамика выпуска и разрыв")
        
        # График 1: Выпуск
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df["date"], y=df["output"],
                                  mode="lines+markers", name="Факт",
                                  line=dict(color="#62C358"),
                                  marker=dict(size=6, line=dict(width=1,color="#62C358"), color="white"),
                                  hovertext=df["quarter_label"]))
        
        fig1.add_trace(go.Scatter(x=df["date"], y=df["output_sa"],
                                  mode="lines+markers", name="Факт (без сезон.)",
                                  line=dict(color="#085800"),
                                  marker=dict(size=6, line=dict(width=1,color="#085800"), color="white"),
                                  hovertext=df["quarter_label"]))

        fig1.add_trace(go.Scatter(x=df["date"], y=df["potential_output_inv"],
                                  mode="lines+markers", name="Потенциальный выпуск",
                                  line=dict(color="#A30008"),
                                  marker=dict(size=6, line=dict(width=1,color="#A30008"), color="white"),
                                  hovertext=df["quarter_label"]))

        # Лента ±2.5%
        fig1.add_trace(go.Scatter(
            x=list(df["date"])+list(df["date"][::-1]),
            y=list(df["potential_output_inv"]*1.025)+list(df["potential_output_inv"]*0.975)[::-1],
            fill="toself",
            fillcolor="rgba(239,124,83,0.2)",
            line=dict(color="transparent"),
            showlegend=True,
            name="±2.5%"
        ))

        fig1.update_layout(title="Динамика выпуска в ценах 2016 года",
                           yaxis_title="Млн/Млрд рублей")

        st.plotly_chart(fig1, use_container_width=True)

        # График 2: Разрыв
        colors_gap = ["#085800" if x >= 0 else "#A30008" for x in df["output_gap_inventory"]]
        fig2 = go.Figure(go.Bar(
            x=df["date"], y=df["output_gap_inventory"], marker_color=colors_gap,
            hovertext=[f"{q}<br>Разрыв: {v:,.0f}".replace(",", " ") 
                       for q,v in zip(df["quarter_label"], df["output_gap_inventory"])]
        ))
        fig2.update_layout(title="Разрыв выпуска", yaxis_title="Млн/Млрд рублей")

        st.plotly_chart(fig2, use_container_width=True)

    # === Вкладка 2: Занятость ===
    with tab2:
        st.subheader("Динамика занятости")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df["date"], y=df["employment"],
                                  mode="lines+markers", name="Факт",
                                  line=dict(color="#085800"),
                                  marker=dict(size=6, line=dict(width=1,color="#085800"), color="white"),
                                  hovertext=df["quarter_label"]))
        fig3.add_trace(go.Scatter(x=df["date"], y=df["potential_employment"],
                                  mode="lines+markers", name="Потенциальная",
                                  line=dict(color="#A30008"),
                                  marker=dict(size=6, line=dict(width=1,color="#A30008"), color="white"),
                                  hovertext=df["quarter_label"]))
        fig3.update_layout(title="Динамика занятости", yaxis_title="Человек")
        st.plotly_chart(fig3, use_container_width=True)

    # === Вкладка 3: Капитал и инвестиции ===
    with tab3:
        st.subheader("Капитал и инвестиции")
        
        # Капитал
        fig4 = px.line(df, x="date", y="capital_inventory", title="Объём капитала в ценах 2016 года")
        fig4.update_traces(mode="lines+markers", line=dict(color="#085800"),
                           marker=dict(size=6, line=dict(width=1,color="#085800"), color="white"),
                           hovertext=df["quarter_label"])
        st.plotly_chart(fig4, use_container_width=True)

        # Инвестиции
        fig5 = px.line(df, x="date", y="investments", title="Инвестиции в основные фонды в ценах 2016 года")
        fig5.update_traces(mode="lines+markers", line=dict(color="#085800"),
                           marker=dict(size=6, line=dict(width=1,color="#085800"), color="white"),
                           hovertext=df["quarter_label"])
        st.plotly_chart(fig5, use_container_width=True)

else:
    st.info("⬆️ Загрузите файл с данными для отображения дашборда")
