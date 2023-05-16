import argparse
import datetime
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import torch

import darts
from darts import TimeSeries, concatenate
from darts.models.forecasting.baselines import NaiveSeasonal
from darts.models import TFTModel
from darts.metrics import mae, mape
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries as dt_attr
from darts.utils.likelihood_models import QuantileRegression

import warnings
warnings.filterwarnings('ignore')

def encode_series(index, series, prefix: str):
    unique = set(series.values)
    columns = {}
    for value in unique:
        columns[f'{prefix}_{value}'] = np.where(series == value, 1., 0.)

    df = pd.DataFrame(data=columns, index=index)
    # print(df.info())
    # print(df.head())
    return df

class TFTForecast:
    def __init__(self, traffic_csv_fn: str, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.df = pd.read_csv(traffic_csv_fn).reset_index()
        self.df.index = pd.to_datetime(self.df.date_time)
        self.df.sort_index(inplace=True)
        self.df = self.df[~self.df.index.duplicated()]

        self.series = TimeSeries.from_dataframe(self.df[['date_time', 'traffic_volume']],
                                                time_col='date_time',
                                                value_cols='traffic_volume',
                                                fill_missing_dates=True,
                                                freq='1h',
                                                )

        cov_dfs = []

        for column in ['weather_main', 'weather_description']:
            df = encode_series(self.df.index, self.df[column], column)
            cov_dfs.append(df)

        cov_df = pd.concat(cov_dfs, axis=1)
        #cov_df = pd.DataFrame()
        cov_df.index = self.df.index
        for column in ['temp', 'rain_1h', 'snow_1h']:
            cov_df[column] = self.df[column]
        cov_df['clouds_all'] = self.df['clouds_all'] / 100.
        cov_df['holiday'] = pd.isna(self.df.holiday).astype(float)

        # print(cov_df.info())
        # print(cov_df.head())
        # print(cov_df.tail())


        self.cov_series = TimeSeries.from_dataframe(cov_df.astype(np.float32),
                                                    fill_missing_dates=True,
                                                    freq='1h',
                                                    )

        self.cov_series = darts.utils.missing_values.fill_missing_values(self.cov_series, fill='auto')
        self.series = darts.utils.missing_values.fill_missing_values(self.series, fill='auto')

        self.validation_size = 24*7*2

        self.train_data = self.series[:-self.validation_size]
        self.test_data = self.series[-self.validation_size:]

        print(f'train: {len(self.train_data)}, test: {len(self.test_data)}')


    def plot_results(self, prefix, true_results, pred_results, dst_file):
        figsize = (9, 8)
        lowest_q, low_q, high_q, highest_q = 0.01, 0.1, 0.9, 0.99
        label_q_outer = f'{int(lowest_q * 100)}-{int(highest_q * 100)}th percentiles'
        label_q_inner = f'{int(low_q * 100)}-{int(high_q * 100)}th percentiles'

        plt.figure(figsize=figsize)
        true_results.plot(label="true")

        # plot prediction with quantile ranges
        pred_results.plot(low_quantile=lowest_q, high_quantile=highest_q, label=label_q_outer)
        pred_results.plot(low_quantile=low_q, high_quantile=high_q, label=label_q_inner)

        mae_result = mae(true_results, pred_results)
        mape_result = mape(true_results, pred_results)

        plt.title(f'{prefix} MAE: {mae_result:.2f}, MAPE: {mape_result:.2f}%')
        plt.legend()
        plt.savefig(dst_file)

    def train_baseline(self):
        naive_seasonal = NaiveSeasonal(K=24*7)
        naive_seasonal.fit(self.train_data)
        pred_naive = naive_seasonal.predict(len(self.test_data))
        mae_naive = mae(self.test_data, pred_naive)
        dst_file = os.path.join(self.output_dir, 'naive_seasonal_baseline.png')
        self.plot_results('naive seasonal baseline', self.test_data, pred_naive, dst_file)
        print(f'naive model: mae: {mae_naive:.2f} -> {dst_file}')

    def train_nbeats(self):
        train_scaler = Scaler()
        scaled_train = train_scaler.fit_transform(self.train_data)
        scaled_val = train_scaler.transform(self.test_data)

        cov = concatenate(
            [
                # dt_attr(self.series.time_index, 'day', dtype=np.float32),
                # dt_attr(self.series.time_index, 'week', dtype=np.float32),

                # dt_attr(self.series.time_index, 'day', dtype=np.float32, one_hot=True),
                dt_attr(self.series.time_index, 'day', dtype=np.float32),
                dt_attr(self.series.time_index, 'day_of_week', dtype=np.float32, one_hot=True),
                dt_attr(self.series.time_index, 'week', dtype=np.float32),
                #dt_attr(self.series.time_index, 'month', dtype=np.float32, one_hot=True),
                dt_attr(self.series.time_index, 'month', dtype=np.float32),
                dt_attr(self.series.time_index, 'year', dtype=np.float32),
                self.cov_series,
            ],
            axis='component'
        )
        cov_scaler = Scaler()
        scaled_cov = cov_scaler.fit_transform(cov)
        quantiles = [
            0.01,
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
            0.99,
        ]
        model = TFTModel(
            input_chunk_length=24*7,
            output_chunk_length=24,

            random_state=42,

            pl_trainer_kwargs={
                "accelerator": "gpu",
                "devices": [0]
            },

            batch_size=128,
            save_checkpoints=True,
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs={'lr': 1e-3},
            lr_scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
            lr_scheduler_kwargs={'threshold': 1e-4, 'verbose': True, 'patience': 10, 'factor': 0.1, 'min_lr': 1e-4},
            nr_epochs_val_period=1,

            hidden_size=128,
            lstm_layers=1,
            num_attention_heads=4,
            dropout=0.2,
            n_epochs=300,
            add_relative_index=False,
            add_encoders=None,
            likelihood=QuantileRegression(
                quantiles=quantiles
            ),
        )

        model.fit(series=scaled_train, val_series=scaled_val, epochs=50, future_covariates=scaled_cov, val_future_covariates=scaled_cov, verbose=True)

        #model = model.load_from_checkpoint(model_name='2023-05-15_13_58_47_torch_model_run_93', best=True)

        scaled_pred_nbeats = model.predict(n=len(self.test_data), future_covariates=scaled_cov)
        pred_nbeats = train_scaler.inverse_transform(scaled_pred_nbeats)
        mae_results = mae(self.test_data, pred_nbeats)

        dst_file = os.path.join(self.output_dir, 'tft.png')
        self.plot_results('TFT', self.test_data, pred_nbeats, dst_file)

        print(f'TFT model: mae: {mae_results:.2f} -> {dst_file}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', required=True, type=str, help='Input csv')
    parser.add_argument('--output_dir', required=True, type=str, help='Output dir')
    FLAGS = parser.parse_args()

    nbeats = TFTForecast(FLAGS.input_csv, FLAGS.output_dir)
    nbeats.train_baseline()
    nbeats.train_nbeats()

if __name__ == '__main__':
    main()
