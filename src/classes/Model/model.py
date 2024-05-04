import random
from typing import Tuple

import numpy as np


class Model:
    def __init__(self):
        self.image = None

    def add_model(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        min_len = min(len(data1), len(data2))
        values1 = data1[:min_len]
        values2 = data2[:min_len]
        return values1 + values2

    def noise(self, N: int, R: float) -> np.ndarray:
        noise_values = np.random.uniform(-R, R, size=N)

        x_min = np.min(noise_values)
        x_max = np.max(noise_values)

        data = ((noise_values - x_min) / (x_max - x_min) - 0.5) * 2 * R

        return data

    def spikes(self, N: int, M: int, R: float, Rs: float) -> Tuple[np.ndarray, dict]:
        data = np.zeros(N)
        positions = random.sample(range(N), M)
        values_emissions_plus = np.random.uniform(R - Rs, R + Rs, size=M)
        values_emissions_minus = np.random.uniform(-R - Rs, -R + Rs, size=M)
        values_emissions = np.concatenate(
            [values_emissions_plus, values_emissions_minus]
        )
        random.shuffle(values_emissions)
        values = values_emissions[:M]
        data[positions] = values

        return data

    def harm(self, N: int, A0: int, f0: int, delta_t: float) -> np.ndarray:
        if delta_t > 1 / (2 * f0):
            return None

        k = np.arange(0, N)
        harm_data = A0 * np.sin(2 * np.pi * f0 * k * delta_t)

        return harm_data

    def descending_exponential_trend(
        self, n: int, a: float, b: float, dt: float
    ) -> np.ndarray:
        k = np.arange(0, n)
        trend_values = b * np.exp(-a * k * dt)
        return trend_values

    def multi_model(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        min_len = min(len(data1), len(data2))
        values1 = data1[:min_len]
        values2 = data2[:min_len]
        return values1 * values2

    def rhythm(self, N, M, R, Rs):
        x_t = [
            random.random() * 2 * Rs + (R - Rs) if i % M == 0 and i != 0 else 0
            for i in range(N)
        ]
        return x_t

    def convolModel(self, first_values, second_values, M):
        N = min(len(first_values), len(second_values))
        first_values = first_values[:N]
        second_values = second_values[:N]

        convolution_data = np.convolve(first_values, second_values)[: N + M - 1]
        convolution_data = convolution_data[M // 2 : -M // 2]

        return convolution_data
