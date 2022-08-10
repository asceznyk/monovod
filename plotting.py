import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from bokeh.models.widgets import Panel, Tabs
from bokeh.io import output_file, show
from bokeh.plotting import figure, ColumnDataSource
from bokeh.layouts import column, layout, gridplot
from bokeh.models import Div, WheelZoomTool
from bokeh.models.widgets import Panel, Tabs


def visualize_paths(pred_path, gt_path=None, html_tile="", title="VO exercises", file_out="plot.html"):
    output_file(file_out, title=html_tile)
    tools = "pan,wheel_zoom,box_zoom,box_select,lasso_select,reset"

    pred_path = np.array(pred_path)
    pred_x, pred_y = pred_path.T

    fig1 = figure(title="Paths", tools=tools, match_aspect=True, width_policy="max", toolbar_location="above", x_axis_label="x", y_axis_label="y")

    if gt_path is not None:
        gt_path = np.array(gt_path)
        gt_x, gt_y = gt_path.T
        xs = list(np.array([gt_x, pred_x]).T)
        ys = list(np.array([gt_y, pred_y]).T)
        diff = np.linalg.norm(gt_path - pred_path, axis=1)
        source = ColumnDataSource(data=dict(gtx=gt_path[:,0], gty=gt_path[:,1],
                                            px=pred_path[:,0], py=pred_path[:,1],
                                            disx=xs, disy=ys))

        fig1.circle("gtx", "gty", source=source, color="blue", hover_fill_color="firebrick", legend_label="GT")
        fig1.line("gtx", "gty", source=source, color="blue", legend_label="GT")
        fig1.multi_line("disx", "disy", source=source, legend_label="Error", color="red", line_dash="dashed")
    else:
        source = ColumnDataSource(data=dict(px=pred_path[:,0], py=pred_path[:,1]))
   
    fig1.circle("px", "py", source=source, color="green", hover_fill_color="firebrick", legend_label="Pred")
    fig1.line("px", "py", source=source, color="green", legend_label="Pred")
    fig1.legend.click_policy = "hide"

    show(layout([Div(text=f"<h1>{title}</h1>"),
                 Div(text="<h2>Paths</h1>"),
                 [fig1],
                 ], sizing_mode='scale_width'))



