from typing import Tuple
import plotly.express as px
import streamlit as st


def plot_fourier_spectrum(data, x_label, y_label, color="blue", title=""):
    fig = px.line(
        x=data["f"], y=data["|Xn|"], labels={"x": x_label, "y": y_label}, title=title
    )
    fig.update_traces(line=dict(color=color, width=2))
    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label, hovermode="x")
    st.plotly_chart(fig, use_container_width=True)


def plot_autocorrelation(data, x_label, y_label, color="blue"):
    fig = px.line(
        x=data.index[1:],
        y=data["AC"][1:],
        labels={"x": x_label, "y": y_label},
    )
    fig.update_traces(line=dict(color=color, width=2))
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode="x",
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_cross_correlation(data, x_label, y_label, color="blue"):
    fig = px.line(
        x=data.index[1:],
        y=data["CCF"][1:],
        labels={"x": x_label, "y": y_label},
    )
    fig.update_traces(line=dict(color=color, width=2))
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode="x",
    )
    st.plotly_chart(fig, use_container_width=True)


def get_harm_value(
    n_value: int = 1000,
    a_value: int = 100,
    f_value: float = 15.0,
    delta_value: float = 0.001,
) -> Tuple[int, int, float, float]:
    st.sidebar.header("Настройка генерации гармонического процесса")
    n_harm = st.sidebar.number_input(
        'Выберите значение "N"', min_value=1, max_value=10000, step=1, value=n_value
    )
    a_harm = st.sidebar.number_input(
        "Выберите значение амплитуды",
        min_value=1,
        max_value=1000,
        step=1,
        value=a_value,
    )
    f_harm = st.sidebar.number_input(
        "Выберите значение частоты",
        min_value=0.1,
        max_value=600.0,
        step=100.0,
        value=f_value,
    )
    delta_t = st.sidebar.number_input(
        'Выберите значение "delta_t"',
        min_value=0.0,
        max_value=0.1,
        step=0.001,
        value=delta_value,
    )

    return n_harm, a_harm, f_harm, delta_t


def plot_line_chart(param1, param2, x_label, y_label, color, width=1):
    fig = px.line(
        x=param1,
        y=param2,
        labels={"x": x_label, "y": y_label},
    )
    fig.update_traces(line=dict(color=color, width=width))
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode="x",
    )
    st.plotly_chart(fig, use_container_width=True)
