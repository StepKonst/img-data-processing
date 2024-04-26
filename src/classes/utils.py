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
