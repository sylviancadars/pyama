from pyama import plotUtils as pu

data_sets_per_subplot = 2
nb_of_cols = 2

for i in range(9):
    nb_of_datasets = i + 1
    nb_of_rows = pu.get_nb_of_subplot_rows(nb_of_datasets, data_sets_per_subplot, nb_of_cols)
    print(100*'*')
    print(f'nb_of_datasets = {nb_of_datasets} ; ({nb_of_rows} rows, {nb_of_cols} columns)')
    print(100*'*')
    for data_set_index in range(nb_of_datasets):
        row_index, col_index, plot_index = pu.get_row_col_and_subplot_indexes(
            data_set_index, nb_of_rows, nb_of_cols, data_sets_per_subplot)
        print(f'data_set # {data_set_index}: ({row_index}, {col_index}), plot_index = {plot_index}')


