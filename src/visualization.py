import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

from evaluation_metrics import EvaluationMetric

import subgroup


class Visualize:
    def __init__(self, eval_metric: EvaluationMetric,
                 dataset: subgroup.Subgroup,
                 subgroups: list[subgroup.Subgroup],
                 target_columns: list[str],
                 translations: dict,
                 cols: int,
                 group_size: int,
                 include_dataset: bool = True,
                 height: int = None):
        """Implements visualization of EMM results.

        Unfortunately,

        Args:
            eval_metric: The evaluation metric used to discover the exceptional
                subgroups.
            dataset: The entire dataset.
            subgroups: List of subgroups that should be visualized.
            target_columns: Name sof the target columns.
            translations: Translations that should be done.
            cols: Number of columns of plots.
            group_size: Size of groupings.
            include_dataset: Whether or not to include the entire dataset in the
                visualization.
            height: Height of each plot. If this is given, then each plot will
                be a 3:2 plot and auto scaling will be turned off.
        """
        self.eval_metric = eval_metric
        self.sets, self.titles = self.setup_sets_titles(include_dataset,
                                                        dataset,
                                                        subgroups,
                                                        group_size)
        self.rows = math.ceil((group_size + 1) / cols)
        self.cols = cols
        self.translations = translations
        self.target_columns = target_columns
        self.height = height
        self.ranges = {"x": [dataset.data[target_columns[0]].min(),
                             dataset.data[target_columns[0]].max()]}

        if len(target_columns) > 1:
            self.ranges['y'] = [dataset.data[target_columns[1]].min(),
                                dataset.data[target_columns[1]].max()]

        if len(target_columns) > 2:
            self.ranges['z'] = [dataset.data[target_columns[2]].min(),
                                dataset.data[target_columns[2]].max()]

    @staticmethod
    def setup_sets_titles(include_dataset, dataset, subgroups, group_size):
        """Sets the sets and titles required by visualizer functions.

        Returns: (tuple) the first element is the sets, the second element is
        the
            titles.
        """
        if include_dataset:
            sets = [dataset] + subgroups[:group_size]
            titles = ['Dataset'] + [str(s.description)
                                    for s in subgroups[:group_size]]
        else:
            sets = subgroups[:group_size]
            titles = [str(s.description) for s in subgroups[:group_size]]

        return sets, titles

    def _heatmap_visualize(self, fig):
        """Visualizer for the heatmap quality measure. Shows a heatmap."""
        if len(self.target_columns) != 2:
            raise ValueError(f'Number of target columns is '
                             f'{self.target_columns}. Must be 2.')

        for i, sub_g in enumerate(self.sets):
            fig.add_trace(
                go.Heatmap(z=sub_g.target,
                           x=self.translations[self.target_columns[1]],
                           y=self.translations[self.target_columns[0]],
                           legendgroup=str(i)),
                row=math.floor(i / self.cols) + 1,
                col=(i % self.cols) + 1
            )

        return fig

    def _one_dim_visualize(self, fig):
        """Visualizes one dimension as a column chart."""
        for i, sub_g in enumerate(self.sets):
            fig.add_trace(
                go.Bar(x=sub_g.data[self.target_columns[0]],
                       y=sub_g.target.values,
                       showlegend=False),
                row=math.floor(i / self.cols) + 1,
                col=(i % self.cols) + 1
            )

        fig.update_xaxes(title=self.target_columns[0])
        fig.update_yaxes(title='frequency')

        return fig

    def _two_dim_visualize(self, fig):
        """Visualizes two dimensions as a scatter chart."""
        for i, subgroup in enumerate(self.sets):
            fig.add_trace(
                go.Scatter(x=subgroup.data[self.target_columns[0]],
                           y=subgroup.data[self.target_columns[1]],
                           mode='markers',
                           showlegend=False),
                row=math.floor(i / self.cols) + 1,
                col=(i % self.cols) + 1
            )

            # Preprocesses data to np arrays with the right shape so that the
            # LinearRegression class can handle it
            x_data = np.array(subgroup.data[self.target_columns[0]])
            x_data = x_data.reshape(-1, 1)
            y_data = np.array(subgroup.data[self.target_columns[1]])
            y_data = y_data.reshape(-1, 1)

            # Does the linear regression fitting
            reg = LinearRegression().fit(x_data, y_data)

            # Gets the lower and upper values of the x-axis to predict so we
            # can plot them
            x_reg = np.array([subgroup.data[self.target_columns[0]].min(),
                              subgroup.data[self.target_columns[0]].max()])
            y_reg = reg.predict(x_reg.reshape(-1, 1)).reshape(1, -1)[0]

            # Actually plots the linear regression
            fig.add_trace(
                go.Scatter(x=x_reg,
                           y=y_reg,
                           mode='lines',
                           showlegend=False),
                row=math.floor(i / self.cols) + 1,
                col=(i % self.cols) + 1
            )

        # Rescale axes
        fig.update_xaxes(range=self.ranges['x'])
        fig.update_yaxes(range=self.ranges['y'])

        # Put the axis label as a title
        fig.update_layout(
            title={
                'text': f"x: {self.target_columns[0]} vs. "
                        f"y: {self.target_columns[1]}",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'})

        return fig

    def _three_four_dim_visualize(self):
        """Visualizes three dimensions as a 3D scatter chart.

        The fourth dimension is the color. More dimensions are at this time
        unavailable.
        """
        # Make a list of lists of dicts for the specifications of the figure.
        # This is a table where each row and column on the figure has a
        # dictionary associated with it.
        specs = []
        for _ in range(self.rows):
            r = [{'type': 'scene'} for _ in range(self.cols)]
            specs.append(r)
        fig = make_subplots(
            rows = self.rows,
            cols=self.cols,
            subplot_titles=self.titles,
            specs=specs
        )

        for i, subgroup in enumerate(self.sets):
            x_data = subgroup.data[self.target_columns[0]]
            y_data = subgroup.data[self.target_columns[1]]
            z_data = subgroup.data[self.target_columns[2]]

            row = math.floor(i / self.cols) + 1
            col = (i % self.cols) + 1
            if len(self.target_columns) == 3:
                marker_style = {'size': 4,
                                'opacity': 0.8}
                fig.add_trace(go.Scatter3d(x=x_data,
                                           y=y_data,
                                           z=z_data,
                                           mode='markers',
                                           marker=marker_style,
                                           showlegend=False),
                              row=row,
                              col=col
                              )
            elif len(self.target_columns) == 4:
                marker_style = {'size': 4,
                                'color': subgroup.data[self.target_columns[3]],
                                'colorscale': 'Viridis',
                                'opacity': 0.8}
                fig.add_trace(go.Scatter3d(x=x_data,
                                           y=y_data,
                                           z=z_data,
                                           mode='markers',
                                           marker=marker_style),
                              row=row,
                              col=col
                              )

            fig.update_layout(scene={'xaxis_title': self.target_columns[0],
                                     'yaxis_title': self.target_columns[1],
                                     'zaxis_title': self.target_columns[2]},
                              row=row,
                              col=col)

        return fig

    def visualize(self):
        if self.height is not None:
            row_heights = [self.height] * self.rows
            col_widths = [int(self.height / 3 * 4)] * self.cols
        else:
            row_heights = None
            col_widths = None
        fig = make_subplots(rows=self.rows,
                            cols=self.cols,
                            subplot_titles=self.titles,
                            row_heights=row_heights,
                            column_width=col_widths)

        if self.eval_metric.name() == "Heatmap":
            fig = self._heatmap_visualize(fig)
        elif len(self.target_columns) == 1:
            fig = self._one_dim_visualize(fig)
        elif len(self.target_columns) == 2:
            fig = self._two_dim_visualize(fig)
        elif len(self.target_columns) > 4:
            raise ValueError("Cannot visualize more than 4 dimensions due to "
                             "the limits of the human experience of space and "
                             "time.")
        else:
            # I hate that this is different and doesn't take fig as an argument,
            # but the way plotly works forces this inconsistency - Yvan
            fig = self._three_four_dim_visualize()

        if self.height is not None:
            fig.update_layout(autosize=False,
                              height=1.3 * self.height * self.rows,
                              width=1.2 * int(self.height / 3 * 4) * self.cols)

        fig.show()
