data_sets_per_subplot = 2
nb_of_cols = 2

def _get_nb_of_rows(len_data_types):
    nb_of_rows = ((len_data_types-1)//(data_sets_per_subplot*nb_of_cols)) + 1
    return nb_of_rows

def _get_row_and_col_indexes(i, nb_of_rows, nb_of_cols):
    col_index = i // (nb_of_rows*data_sets_per_subplot)
    plot_index = i // data_sets_per_subplot
    row_index = plot_index - col_index * nb_of_rows
    return col_index, row_index, plot_index

for i in range(16):
    len_data_types = i + 1
    nb_of_rows = _get_nb_of_rows(len_data_types)
    print(f'len_data_types = {len_data_types} ; ({nb_of_rows} rows, {nb_of_cols} columns)')

print(100*'*')
len_data_types = 16
nb_of_rows = _get_nb_of_rows(len_data_types)
for i in range(len_data_types):
    col_index, row_index, plot_index = _get_row_and_col_indexes(i, nb_of_rows, nb_of_cols)
    print(f'{i}: ({row_index}, {col_index}), plot_index = {plot_index}')

