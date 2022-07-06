# Comments guidelines:

##### [HEADING 1] #####
### [HEADING 2] ###
# Regular comment
# END [HEADING 2] #
#-- END [HEADING 1] --#

# --------------------

##### IMPORTS #####
from bokeh.plotting import figure, curdoc 
from bokeh.models import ColumnDataSource, ColorBar, FixedTicker, BasicTickFormatter, DataTable, TableColumn, HoverTool
from bokeh.transform import linear_cmap, log_cmap
from bokeh.palettes import Oranges9
from bokeh.layouts import row, column
import colorcet as cc # https://colorcet.holoviz.org/user_guide/index.html#complete-list
import pickle
import os
import pandas as pd
import numpy as np
#-- END IMPORTS --#

##### LOAD DATA #####
base_data_path = 'disentanglement_data'
dataset_name = 'beta_vae_40'
data_path = os.path.join(base_data_path, dataset_name)

# Scaler
qt_labels = pickle.load(open(os.path.join(data_path, 'label_scaler.pkl'), 'rb'))

# Wavelengths
wavelengths = np.load(os.path.join(data_path, 'wavelengths.npy'))

# Dataframe
dis_results_df = pd.read_pickle(os.path.join(data_path, 'dis_results.gz'))
#-- END LOAD DATA --#

##### PREPARE DATA #####
### INVERSE TEMPERATURES ###
_mock_labels = np.array([0.81681682, 0.55005005, 0.63313313, 0.23323323, 0.43343343,
       0.42842843, 1.        , 0.43343343, 0.63313313, 0.7997998 ,
       0.8998999 , 0.63313313, 0.43343343, 0.16666667, 0.36636637,
       0.53553554, 0.53353353, 0.2997998 , 0.43343343, 0.5       ,
       0.5       , 0.        , 0.        , 0.        , 0.        ,
       0.1001001 ])

i_original_teff_list = list()
i_new_teff_list = list()

for _, row_ in dis_results_df.iterrows():
    # Original teff
    _mock_labels[0] = row_.original_teff
    _teff_value = qt_labels.inverse_transform(np.array([_mock_labels]))[0,0]
    i_original_teff_list.append(_teff_value)
    
    # New teff
    _mock_labels[0] = row_.new_teff
    _teff_value = qt_labels.inverse_transform(np.array([_mock_labels]))[0,0]
    i_new_teff_list.append(_teff_value)
    
dis_results_df['i_original_teff'] = i_original_teff_list
dis_results_df['i_new_teff'] = i_new_teff_list

# Cast teffs to str
dis_results_df['i_original_teff'] = dis_results_df['i_original_teff'].astype(str)
dis_results_df['i_new_teff'] = dis_results_df['i_new_teff'].astype(str)
# END INVERSE TEMPERATURES #

# Group temperatures to calculate disentanglement results by temperature
average_disentangled_results_df = dis_results_df.groupby(by=['i_original_teff', 'i_new_teff']).mean().reset_index()
#-- END PREPARE DATA --#

##### TOOLS AND TOOLTIPS #####
GRID_TOOLS = ['tap', 'reset']
SPECTRA_TOOLS = ['pan', 'box_zoom', 'reset']

GRID_TOOLTIPS = [
    ("index", "$index"),
    ("original_teff", "@i_original_teff"),
    ("generated_teff", "@i_new_teff"),
    ("error", "@mse{(0.000)}")
]
#-- END TOOLS AND TOOLTIPS --#

##### BOKEH FIGURES #####
### GRID FIGURE ###
grid_df = average_disentangled_results_df.copy(deep=True)

extended_source = ColumnDataSource(data=dis_results_df) # TODO I don't know if this is useful...
source = ColumnDataSource(data=grid_df)

palette = list(Oranges9); palette.reverse()
#palette = cc.CET_L18

mapper = linear_cmap(field_name='mse', palette=palette, low=grid_df.mse.min(), 
                           high=grid_df.mse.max())

x_range = np.unique(grid_df.i_original_teff)
y_range = np.unique(grid_df.i_new_teff)

grid_figure = figure(title="Disentanglement",
           x_range=x_range,
           y_range=y_range,
           tools=GRID_TOOLS,
           tooltips=GRID_TOOLTIPS,
           width=700, height=700)


grid_figure.rect(x="i_original_teff", y="i_new_teff", #width=.0004, height=.025,
       width=1, height=1,
       source=source,
       fill_color=mapper,
       line_color=None)

grid_figure.xaxis.major_label_orientation = np.pi/2
grid_figure.grid.grid_line_color = None
grid_figure.axis.axis_line_color = None
grid_figure.axis.major_tick_line_color = None
grid_figure.axis.major_label_text_font_size = "10px"
grid_figure.xaxis.axis_label = 'Original Teff'
grid_figure.yaxis.axis_label = 'Generated Teff'

# Color Bar
n_ticks = 10  # how many ticks do you want?
ticks = np.linspace(grid_df.mse.min(), grid_df.mse.max(), n_ticks)  # round to desired precision 
color_ticks = FixedTicker(ticks=ticks)

color_bar = ColorBar(color_mapper=mapper['transform'],
                     ticker=color_ticks,
                     formatter=BasicTickFormatter(precision=2, use_scientific=False))
grid_figure.add_layout(color_bar, 'right')
# END GRID FIGURE #

### DATATABLE FIGURE ###
table_source = ColumnDataSource(data=dict({
    'object_id': np.array([]),
    'original_teff': np.array([]),
    'new_teff': np.array([]),
    'mse': np.array([])
}))

table_columns = [
        TableColumn(field="object_id", title="Object ID"),
        TableColumn(field="original_teff", title="original_teff"),
        TableColumn(field="new_teff", title="new_teff"),
        TableColumn(field="mse", title="MSE"),
    ]
data_table = DataTable(source=table_source, columns=table_columns) #, width=400, height=280)
# END DATATABLE FIGURE #

### SPECTRA COMPARISON FIGURE ###
spectra_fig = figure(title="Spectra Comparison",
                     tools=SPECTRA_TOOLS,
                     y_range=(-0.1,1.1)
                    )


orig_spectrum_source = ColumnDataSource(data=dict({
    'spectra': np.array([]),
    'waves': np.array([]),
}))

new_spectrum_source = ColumnDataSource(data=dict({
    'spectra': np.array([]),
    'waves': np.array([]),
}))

abs_diff_source = ColumnDataSource(data=dict({
    'abs_diff': np.array([]),
    'zero': np.array([]),
    'waves': np.array([]),
}))

spectra_fig.varea(x="waves", y1='zero', y2="abs_diff", source=abs_diff_source,
                  #color='firebrick', 
                  alpha=0.15,
                  legend_label='error')

spectra_fig.line(x="waves", y="spectra", source=orig_spectrum_source,
                 line_width=1.5, line_dash=(4,4),
                 line_color='gray',
                 legend_label='Original spectrum')

spectra_fig.line(x="waves", y="spectra", source=new_spectrum_source, 
                 line_width=1.5, 
                 line_color='orange',
                 legend_label='New spectrum')

# END SPECTRA COMPARISON FIGURE #

### LAYOUT CONFIGURATION ###
layout = row(grid_figure, 
         column(data_table, spectra_fig, sizing_mode='stretch_both'), 
         #data_table,
         sizing_mode='stretch_both')
# END LAYOUT CONFIGURATION #
#-- END BOKEH FIGURES --#


##### CALLBACKS #####
def update_table(attr, old, new):
    
    # If there is any cell selected we empty the datatable
    if (len(new) == 0): 
        table_source.data = {
            'object_id': np.array([]),
            'original_teff': np.array([]),
            'new_teff': np.array([]),
            'mse': np.array([])
        }

        # We also have to empty the figure about spectra
        orig_spectrum_source.data = {
            'spectra': np.array([]),
            'waves': np.array([]),
        }

        new_spectrum_source.data = {
            'spectra': np.array([]),
            'waves': np.array([]),
        }

        abs_diff_source.data = {
            'abs_diff': np.array([]),
            'zero': np.array([]),
            'waves': np.array([]),
        }

        return
    
    _filter_df = dis_results_df[(dis_results_df['i_original_teff'] == source.data['i_original_teff'][new][0]) &
                              (dis_results_df['i_new_teff'] == source.data['i_new_teff'][new][0])]

    # Select data and update table source
    _obj_ids = _filter_df['ids'].values
    _original_teffs = _filter_df['i_original_teff'].values
    _new_teffs = _filter_df['i_new_teff'].values
    _mses = _filter_df['mse'].values

    table_source.data = {
        'object_id': np.array(_obj_ids),
        'original_teff': np.array(_original_teffs),
        'new_teff': np.array(_new_teffs),
        'mse': np.array(_mses)
    }

    # We empty the selected rows in the table source
    table_source.selected.indices = []

    return

def update_spectra(attr, old, new):

    # If there are not any row selected, we empty the figure. It should only happens when the table source is updated with a selected row (because we empty it manually)
    if (len(new) == 0):
        orig_spectrum_source.data = {
            'spectra': np.array([]),
            'waves': np.array([]),
        }
        new_spectrum_source.data = {
            'spectra': np.array([]),
            'waves': np.array([]),
        }
        abs_diff_source.data = {
            'abs_diff': np.array([]),
            'zero': np.array([]),
            'waves': np.array([]),
        }
        return
    
    _filter_df = dis_results_df[(dis_results_df['i_original_teff'] == table_source.data['original_teff'][new][0]) &
                                (dis_results_df['i_new_teff'] == table_source.data['new_teff'][new][0]) &
                                (dis_results_df['ids'] == table_source.data['object_id'][new][0])]

    # Select data, transform to list(necessary to multi_line) and update spectra source
    #_original_spectra = np.vstack(_filter_df['original_spectra'].values).tolist()
    #_new_spectra = np.vstack(_filter_df['new_spectra'].values).tolist()
    _original_spectra = _filter_df['original_spectra'].values[0]
    _new_spectra = _filter_df['new_spectra'].values[0]

    #_abs_diff = np.power(_original_spectra-_new_spectra, 2)
    _abs_diff = np.abs(_original_spectra - _new_spectra)

    orig_spectrum_source.data = {
        'spectra': _original_spectra,
        'waves': wavelengths,
    }
    new_spectrum_source.data = {
        'spectra': _new_spectra,
        'waves': wavelengths,
    }
    abs_diff_source.data = {
        'abs_diff': _abs_diff,
        'zero': np.zeros(len(wavelengths)),
        'waves': wavelengths,
    }
    return

source.selected.on_change('indices', update_table)

table_source.selected.on_change('indices', update_spectra)
#-- END CALLBACKS --#

curdoc().add_root(layout)
