# Comments guidelines:

##### [HEADING 1] #####
### [HEADING 2] ###
# Regular comment
# END [HEADING 2] #
#-- END [HEADING 1] --#

# --------------------

##### IMPORTS #####
from bokeh.plotting import figure, curdoc 
from bokeh.models import ColumnDataSource, Rect, ColorBar, FixedTicker, BasicTickFormatter, DataTable, TableColumn, Slider, RadioButtonGroup
from bokeh.transform import linear_cmap, log_cmap
from bokeh.palettes import Oranges9
from bokeh.layouts import row, column
#import colorcet as cc # https://colorcet.holoviz.org/user_guide/index.html#complete-list
import pickle
import os
import pandas as pd
import numpy as np
#-- END IMPORTS --#

ALPHA_ERROR_VALUE = 0.15

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

linear_mapper = linear_cmap(field_name='mse', palette=palette, low=grid_df.mse.min(), 
                            high=grid_df.mse.max())

log_mapper = log_cmap(field_name='mse', palette=palette, low=grid_df.mse.min(), 
                      high=grid_df.mse.max())

DEFAULT_MAPPER = linear_mapper

x_range = np.unique(grid_df.i_original_teff)
y_range = np.unique(grid_df.i_new_teff)

grid_figure = figure(title="Disentanglement Grid",
           x_range=x_range,
           y_range=y_range,
           tools=GRID_TOOLS,
           tooltips=GRID_TOOLTIPS,
           output_backend="webgl")


grid_gr = grid_figure.rect(x="i_original_teff", y="i_new_teff", #width=.0004, height=.025,
                           width=1, height=1,
                           source=source,
                           fill_color=DEFAULT_MAPPER,
                           line_color=None)

# Initialize selected and nonselected sources to be able to color them (https://docs.bokeh.org/en/latest/docs/user_guide/styling.html#selected-and-unselected-glyphs)
selected_grid = Rect(fill_alpha=1, fill_color=DEFAULT_MAPPER, line_color=None)
nonselected_grid = Rect(fill_alpha=0.4, fill_color=DEFAULT_MAPPER, line_color=None)

grid_figure.xaxis.major_label_orientation = np.pi/2
grid_figure.grid.grid_line_color = None
grid_figure.axis.axis_line_color = None
grid_figure.axis.major_tick_line_color = None
grid_figure.axis.major_label_text_font_size = "10px"
grid_figure.xaxis.axis_label = 'Original Teff'
grid_figure.yaxis.axis_label = 'Generated Teff'

grid_gr.selection_glyph = selected_grid
grid_gr.nonselection_glyph = nonselected_grid

# Color Bar
n_ticks = 10  # how many ticks do you want?
ticks = np.linspace(grid_df.mse.min(), grid_df.mse.max(), n_ticks)  # round to desired precision 
color_ticks = FixedTicker(ticks=ticks)

color_bar = ColorBar(color_mapper=DEFAULT_MAPPER['transform'],
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
data_table = DataTable(source=table_source, columns=table_columns,
                       sizing_mode='stretch_width', height=200)
# END DATATABLE FIGURE #

### SPECTRA COMPARISON FIGURE ###
spectra_fig = figure(title="Spectra Comparison",
                     tools=SPECTRA_TOOLS,
                     y_range=(-0.1,1.1),
                     output_backend="webgl"
                    )


orig_spectrum_source = ColumnDataSource(data=dict({
    'spectra': np.array([]),
    'waves': np.array([]),
}))

new_spectrum_source = ColumnDataSource(data=dict({
    'spectra': np.array([]),
    'waves': np.array([]),
}))

spectra_diff_source = ColumnDataSource(data=dict({
    'abs_diff': np.array([]),
    'zero': np.array([]),
    'waves': np.array([]),
}))

varea = spectra_fig.varea(x="waves", y1='zero', y2="abs_diff", source=spectra_diff_source,
                  #color='firebrick', 
                  alpha=ALPHA_ERROR_VALUE,
                  legend_label='Error')

spectra_fig.line(x="waves", y="spectra", source=orig_spectrum_source,
                 line_width=1.5, line_dash=(4,4),
                 line_color='gray',
                 legend_label='Original spectrum')

spectra_fig.line(x="waves", y="spectra", source=new_spectrum_source, 
                 line_width=1.5, 
                 line_color='orange',
                 legend_label='New spectrum')

# END SPECTRA COMPARISON FIGURE #

### WIDGETS ###
alpha_slider = Slider(start=0, end=1, value=ALPHA_ERROR_VALUE, step=0.05, title='Error Alpha Value',
                      width=200, 
                      height=35, 
                      sizing_mode='fixed', 
                      align='end', margin=(5,50,5,50))

LABELS = ['Absolute', 'Integral']
error_radio_button_group = RadioButtonGroup(labels=LABELS, active=0,
                                            width=75,
                                            height=35, 
                                            sizing_mode='fixed',
                                            margin=(5,50,5,5)
                                           )

COLOR_LABELS = ['Linear', 'Log']
color_radio_button_group = RadioButtonGroup(labels=COLOR_LABELS, active=0,
                                            width=75,
                                            height=35, 
                                            align='end',
                                            sizing_mode='fixed',
                                            margin=(5,130,0,5)
                                           )

# END WIDGETS #

### LAYOUT CONFIGURATION ###
layout = row(column(color_radio_button_group, grid_figure, sizing_mode='stretch_both'),
             column(data_table, 
                    column(row(error_radio_button_group, alpha_slider, sizing_mode='stretch_width', align=('end', 'center')), 
                           spectra_fig, 
                           sizing_mode='stretch_both'), 
                    sizing_mode='stretch_both'), 
             sizing_mode='stretch_both')
# END LAYOUT CONFIGURATION #
#-- END BOKEH FIGURES --#


##### CALLBACKS #####
### AUXILIARY FUNCTIONS ###
def update_spectra_diff_source(active, original_spectra, new_spectra):
    
    #_abs_diff = np.power(_original_spectra-_new_spectra, 2)
    _abs_diff = np.abs(original_spectra - new_spectra)

    _active_label = LABELS[active]

    if (_active_label == LABELS[0]): 
        spectra_diff_source.data = {
            'abs_diff': _abs_diff,
            'zero': np.zeros(len(wavelengths)),
            'waves': wavelengths,
        } 
    elif (_active_label == LABELS[1]):
        spectra_diff_source.data = {
            'abs_diff': original_spectra,
            'zero': new_spectra,
            'waves': wavelengths,
        }


# END AUXILIARY FUNCTIONS #
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

        spectra_diff_source.data = {
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
        spectra_diff_source.data = {
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


    orig_spectrum_source.data = {
        'spectra': _original_spectra,
        'waves': wavelengths,
    }
    new_spectrum_source.data = {
        'spectra': _new_spectra,
        'waves': wavelengths,
    }
    
    # Update error in the spectra figure
    update_spectra_diff_source(active=error_radio_button_group.active, original_spectra=_original_spectra,
                               new_spectra=_new_spectra)

    return

def my_slider_handler(attr, old, new):
    varea.glyph.fill_alpha = new

def my_error_rbg_handler(attr, old, new):
    _indices = table_source.selected.indices

    if (len(_indices) == 0):
        return

    _filter_df = dis_results_df[(dis_results_df['i_original_teff'] == table_source.data['original_teff'][_indices][0]) &
                                (dis_results_df['i_new_teff'] == table_source.data['new_teff'][_indices][0]) &
                                (dis_results_df['ids'] == table_source.data['object_id'][_indices][0])]

    # Select data, transform to list(necessary to multi_line) and update spectra source
    #_original_spectra = np.vstack(_filter_df['original_spectra'].values).tolist()
    #_new_spectra = np.vstack(_filter_df['new_spectra'].values).tolist()
    _original_spectra = _filter_df['original_spectra'].values[0]
    _new_spectra = _filter_df['new_spectra'].values[0]

    update_spectra_diff_source(active=new, original_spectra=_original_spectra,
                               new_spectra=_new_spectra)

def my_color_rbg_handler(attr, old, new):
    if (COLOR_LABELS[new] == 'Linear'):
        grid_gr.glyph.fill_color = linear_mapper
        grid_gr.selection_glyph.fill_color = linear_mapper
        grid_gr.nonselection_glyph.fill_color = linear_mapper
        
    elif (COLOR_LABELS[new] == 'Log'):
        grid_gr.glyph.fill_color = log_mapper
        grid_gr.selection_glyph.fill_color = log_mapper
        grid_gr.nonselection_glyph.fill_color = log_mapper


source.selected.on_change('indices', update_table)

table_source.selected.on_change('indices', update_spectra)

alpha_slider.on_change("value", my_slider_handler)

error_radio_button_group.on_change('active', my_error_rbg_handler)

color_radio_button_group.on_change('active', my_color_rbg_handler)
#-- END CALLBACKS --#

curdoc().add_root(layout)
