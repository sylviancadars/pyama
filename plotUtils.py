# -*- coding: utf-8 -*-
"""
A function to modify

@author: cadarp02
"""
import matplotlib.pyplot as plt
import matplotlib.figure
from tkinter import *
from tkinter.filedialog import asksaveasfile

def changeAxisProperties(fig : matplotlib.figure.Figure,axisIndex:int=0,
                         **kwargs) :
    """
    A function to change the axes properties of a figure with its handle

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure handle.
    axisIndex : int, optional
        Index of the axis system to modify. The default is 0.
    **kwargs : keyward arguments
        Can be any matplotlib property


    Returns
    -------
    fig : matplotlib.figure.Figure
        Modified figure handle.
    """

    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
    ax = fig.get_axes()
    ax[axisIndex].set(**kwargs)
    plt.show()

    return fig

def raise_above_all(window):
    window.attributes('-topmost', True)
    window.attributes('-topmost', False)

def saveFigAsSVG(fig : matplotlib.figure.Figure,fileName:str='',
                 windowTitle:str='Save figure as SVG')-> str :
    """
    A function to export a figure in SVG format in interactive or silent mode

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Handle of the figure to export in SVG format.
    fileName : str, optional
        If the default ('') is used the function runs in interactive mode by
        means of a tkinter popupwindow.
        The default is ''.

    Returns
    -------
    fileName : str
        Name of the file in which figure has been saved.
    """

    if fileName == '' :
        data = [('SVG (*.svg)', '*.svg')]
        print('***\nSave as window \"'+windowTitle+'\" waiting for user action.\n***')
        file = asksaveasfile(filetypes = data, defaultextension = data,
                             title=windowTitle)
        if file != None :
            fileName = file.name
            fig.savefig(fileName,format='svg')
        else :
            print('Save figure operation cancelled.')
        file.close()

    else :
        fig.savefig(fileName,format='svg')
    return fileName

def changeTickLabelsFontSize(fig,tickLabelsFontSize:int,
                             axisIndex:int=0)->matplotlib.figure.Figure :
    """
    A function to change the tick-labels font size of a pyplot figure object

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Handle of the figure to export in SVG format.
    tickLabelsFontSize:int : int
        Tick label font size
    axisIndex:int (default : 0)
        index of axes whose tick-label font size should be modified.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure  hande with modified tick-labels font size.
    """

    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
    ax = fig.get_axes()

    if tickLabelsFontSize > 0 :
        for item in (ax[axisIndex].get_xticklabels() + ax[axisIndex].get_yticklabels()):
            item.set_fontsize(tickLabelsFontSize)
    plt.show()

    return fig


def addVerticalLines(fig : matplotlib.figure.Figure, line_positions,
                     axisIndex:int=0, linestyle='dashed',
                     marker=None, color='k')->matplotlib.figure.Figure :
    print(axesIndex)
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
    ax = fig.get_axes()
    ylim = ax[axisIndex].get_ylim()
    for pos in linepositions:
        ax[axisIndex].plot(2*[pos], [ylim], linestyle=linestyle, marker=maker,
                color=color)
    ax[axisIndex].set(ylim=ylim)
    plt.show()

    return fig

data_sets_per_subplot = 2
nb_of_cols = 2

def get_nb_of_subplot_rows(nb_of_datasets, data_sets_per_subplot=1, nb_of_cols=1):
    """
    Calculate the number of subplot rows from nb of datasets, subplot columns, and datasets per plot
    """
    nb_of_rows = ((nb_of_datasets-1)//(data_sets_per_subplot*nb_of_cols)) + 1
    return nb_of_rows

def get_row_col_and_subplot_indexes(dataset_index, nb_of_rows=1, nb_of_cols=1,
                                           data_sets_per_subplot=1):
    col_index = dataset_index // (nb_of_rows*data_sets_per_subplot)
    subplot_index = dataset_index // data_sets_per_subplot
    row_index = subplot_index - col_index * nb_of_rows
    return row_index, col_index, subplot_index

