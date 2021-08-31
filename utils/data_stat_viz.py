import json
from future.utils import viewitems
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import cv2
import io
import sklearn
import torch
from typing import List, Dict, Any, Tuple
import numpy as np
import logging
import os
import tqdm
from scipy.ndimage import rotate
from pyquaternion import Quaternion
import glob
import pandas as pd
from future.utils import viewitems
from tabulate import tabulate

from plotly.graph_objs import Scatter, Scatter3d, Figure, Layout
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly
import plotly.tools as tls
import plotly.express as px

from typing import List

default_config = {}

class StatViz(object):

    def __init__(self, config={}):
        self.config = default_config
        self.config.update(config)

    def plot_radar(self, data_dict: dict={}):
        '''
        data_dict = {
            'categories': [...],
            'model_1': [<trace1>],
            'model_2': [<trace2>],
        }
        '''
        color_list = ['red', 'green', 'cyan', 'yellow', 'purple']
        categories = data_dict['categories']
        fig = go.Figure()
        i = 0
        for k, v in data_dict.items():
            if k != "categories":
                # plot individual radar
                df = pd.DataFrame(dict(
                    r=v,
                    theta=categories
                ))
                fig_individual = px.line_polar(df, r='r', theta='theta', line_close=True)
                fig_individual.update_traces(fill='toself')
                fig_individual.update_layout(
                    title=k,
                    font=dict(
                        size=18
                    ),
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0,3]
                        )
                    ),

                    showlegend=False
                )

                fig_individual.show()

                fig.add_trace(go.Scatterpolar(
                    r=v,
                    theta=categories,
                    fill='toself',
                    #fillcolor=color_list[i],
                    name=k,
                    opacity=0.5
                ))
                i += 1

        fig.update_layout(
            font=dict(
                size=18
            ),       
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0,3]
                )
            ),
            showlegend=True
        )

        return fig
        
    def plot_metric_distribution(self, df: pd.DataFrame, metrics: list=['ade', 'fde', 'max_dist'], bin_size: list = [0.1, 0.1, 0.1]):

        for metric in metrics:
            fig = px.histogram(df,
                               x=metric,
                               color='model',
                               marginal='box',
                               nbins=20,
                               #barmode='overlay',
                               barmode='group',
                               histnorm='percent',
                               histfunc='count')
            fig.update_layout(font=dict(size=18), yaxis=dict(title='Percent'))
            fig.show()


        hist_stat_dict = {
            'model': [],
        }

        for metric in metrics:
            hist_stat_dict[metric+"_min"] = []
            hist_stat_dict[metric+"_mean"] = []
            hist_stat_dict[metric+"_max"] = []
            hist_stat_dict[metric+"_90_quantile"] = []
            hist_stat_dict[metric+"_std"] = []
            hist_stat_dict[metric+"_median"] = []



            
        model_list = df.model.unique()
        for model in model_list:
            hist_stat_dict['model'].append(model)
            for metric in metrics:
                metric_array = np.array(df.loc[df['model']==model][metric].tolist())
                hist_stat_dict[metric+"_mean"].append(metric_array.mean())
                hist_stat_dict[metric+"_max"].append(metric_array.max())
                hist_stat_dict[metric+"_min"].append( metric_array.min())
                hist_stat_dict[metric+"_std"].append(metric_array.std())
                hist_stat_dict[metric+"_median"].append(np.median(metric_array))
                hist_stat_dict[metric+"_90_quantile"].append(np.quantile(metric_array,0.9, interpolation='nearest'))


        hist_stat_df = pd.DataFrame(hist_stat_dict)
        #print(hist_stat_df)

        for m in metrics:
            m_df = hist_stat_df.loc[:, hist_stat_df.columns.str.contains(m) | hist_stat_df.columns.str.contains('model')]
            #m_df = hist_stat_df.filter(regex=m)
            print(tabulate(m_df, headers='keys', tablefmt='psql'))
            
    def plot_boxplot(self, df_dict: dict, metrics: list, xlabel:str='% Training data removed'):
        group_dict = {
            'model': [],
            xlabel: []
        }
        for m in metrics:
            group_dict[m] = []
        
        for experiment_name, v in viewitems(df_dict):
            for model_name, df in v.items():
                group_dict['model'] += df['model'].to_list()
                group_dict[xlabel] += [experiment_name]*len(df['ade'].to_list())

                for m in metrics:
                    group_dict[m] += df[m].to_list()

        tmp_df = pd.DataFrame(group_dict)
        tmp_df.sort_values(by=xlabel, inplace=True)

        for metric in metrics:
            fig = px.box(tmp_df, x=xlabel, y=metric, color='model')
            fig.update_traces(quartilemethod="inclusive")
            fig.update_layout(
                height=500,
                width=1000,
                font=dict(size=28),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.,
                    xanchor="right",
                    x=1
                )
            )
            fig.show()


    def plot_training_curves(self, df_dict:dict, stddev: int=1, plot_type:str='box'):
        if plot_type == 'box':
            self.plot_training_curves_box(df_dict)
        elif plot_type == 'dist':
            self.plot_training_curves_dist(df_dict, stddev=stddev)
        
    def plot_training_curves_box(self, df_dict:dict, metrics=['ade', 'fde', 'max_dist']):
        group_dict = {
            'ade': [],
            'fde': [],
            'max_dist': [],
            'model': [],
            'epoch': []
        }            
        
        for model_name, v in viewitems(df_dict):
            df_pkl_dir = v['df_pkl_dir']

            for df_path in glob.glob(df_pkl_dir+"/*.pkl"):
                filename = df_path.split("/")[-1]
                df = pd.read_pickle(df_path)
                epoch = int(filename[6:-4])

                nb_samples_in_val = len(df['ade'].to_list())
                
                group_dict['model'] += [model_name] * nb_samples_in_val
                group_dict['epoch'] += [epoch] * nb_samples_in_val

                group_dict['ade'] += [ade[0] for ade in df['ade'].to_list()]
                group_dict['fde'] += [fde[0] for fde in df['fde'].to_list()]
                group_dict['max_dist'] += [max_dist[0] for max_dist in df['max_dist'].to_list()]

                # print(f"{model_name}, {epoch}")
                # print(f"{len(group_dict['model'][-1])}, {len(group_dict['model'][-1])}, {len(group_dict['epoch'][-1])}, {len(group_dict['ade'][-1])}, {len(group_dict['fde'][-1])}, {len(group_dict['max_dist'][-1])}")

        tmp_df = pd.DataFrame(group_dict)
        tmp_df.sort_values(by='epoch', inplace=True)
        for metric in metrics:
            fig = px.box(tmp_df, x='epoch', y=metric, color='model')
            fig.update_traces(quartilemethod="inclusive")
            fig.update_layout(font=dict(size=18))
            fig.show()

                
    def plot_training_curves_dist(self,
                                  df_dict:dict,
                                  distribution=True,
                                  stddev: int=1,
                                  metrics:list=['ade', 'fde', 'max_dist'],
                                  color_list: list = [(255,0,0), (0, 255, 0), (0,0,255)]):

        '''
        df_dict = {<model name>: {"df_pkl_dir":<df pkl dir>, "color": (255,0,0)}
        '''
        
        #### process data ####
        processed_data = {}
        for model_name, v in viewitems(df_dict):
            df_pkl_dir = v['df_pkl_dir']
            d = {
                'epoch': [],
                'ade': [],
                'fde': [],
                'max_dist': [],
                'color': v['color']
            }

            for df_path in glob.glob(df_pkl_dir+"/*.pkl"):
                filename = df_path.split("/")[-1]
                df = pd.read_pickle(df_path)
                #print(df)
                d['epoch'].append(int(filename[6:-4]))
                for metric in metrics:
                    metric_list = df[metric].to_list()
                    probs = df['probabilities'].to_list()
                    metric_list = [m[np.argmax(prob)] for m, prob in zip(metric_list, probs)]
                    d[metric].append(metric_list)
                
            processed_data[model_name] = d
                
        
        #### plot ####
        plotly_data = {
            'ade': [],
            'fde': [],
            'max_dist': []
        }

        y_max = {
            'ade': -1e4,
            'fde': -1e4,
            'max_dist': -1e4,
        }
        y_min = {
            'ade': 1e4,
            'fde': 1e4,
            'max_dist': 1e4,
        }
        for metric in metrics:
            for model_name, model_stat in viewitems(processed_data):
                d = np.array(model_stat[metric])
                mean = np.mean(d, axis=1)
                std = np.std(d, axis=1)
                upper = mean + stddev * std
                lower = mean - stddev * std
            
                x = np.array(model_stat['epoch'])
                x = x - np.min(x)
                
                # sort
                idx = np.argsort(x)
                x = x[idx]
                mean = mean[idx]
                upper = upper[idx]
                lower = lower[idx]
            
                if np.max(upper) > y_max[metric]:
                    y_max[metric] = np.max(upper)
                if np.min(lower) < y_min[metric]:
                    y_min[metric] = np.min(lower)

                    
                upper_trace, lower_trace, mean_trace = self._get_traces(x,
                                                                        upper,
                                                                        lower,
                                                                        mean,
                                                                        model_stat['color'],
                                                                        legend=model_name,
                                                                        spread_interval=None)
                
                plotly_data[metric] += [upper_trace, lower_trace, mean_trace]

        #### plot ####
        for k, v in viewitems(plotly_data):
            layout = go.Layout(
                xaxis=dict(range=[0, np.max(x)],
                           showgrid=True,
                           title='Epoch',
                           titlefont=dict(size=18)),
                yaxis=dict(range=[y_min[k]- 0.1 * y_min[k], y_max[k]+0.1*y_max[k]],
                           showgrid=True,
                           title=k,
                           titlefont=dict(size=18)),
                legend=dict(orientation='h', x=0, y=1.15, font=dict(size=15))
            )
            fig = go.Figure(data=v, layout=layout)
            fig.show()


    def get_latex_result_table(self, data_dir: str):
        pass

    def _get_traces(self,
                   x,
                   upper_traj,
                   lower_traj,
                   mean_traj,
                   color_scale,
                   legend,
                   spread_interval=None):

        if spread_interval:
            idx = np.where(x % spread_interval == 0)
            x = x[idx]
            upper_traj = upper_traj[idx]
            lower_traj = lower_traj[idx]
            mean_traj = mean_traj[idx]
        
        upper_trace = go.Scatter(
            x = x,
            y=upper_traj,
            fill=None,
            mode='lines',
            line=dict(
                color='rgb'+str(color_scale),
            ),
            showlegend=False,
            name=legend
        )

        lower_trace = go.Scatter(
            x = x,
            y=lower_traj,
            fill="tonexty",
            mode='lines',
            line=dict(
                color='rgb'+str(color_scale),
            ),
            showlegend=False,
            name=legend
        )
        
        mean_trace = go.Scatter(
            x = x,
            y=mean_traj,
            mode='lines',
            line=dict(
                #color='rgb('+str(color_scale)+','+str(color_scale)+','+ str(color_scale)+')',
                color='rgb'+str(color_scale),
            ),
            name=legend)
        
        return upper_trace, lower_trace, mean_trace

