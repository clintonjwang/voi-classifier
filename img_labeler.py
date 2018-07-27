import pandas as pd
import numpy as np
import easygui
import sys
import glob
from os.path import *
"""import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.backends.tkagg as tkagg
from matplotlib.backends.backend_agg import FigureCanvasAgg
from skimage.transform import rescale"""

#  Inspired by matplotlib source: lib/matplotlib/backends/backend_tkagg.py
"""def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasAgg(figure)
    figure_canvas_agg.draw()
    figure_x, figure_y, figure_w, figure_h = figure.bbox.bounds
    figure_w, figure_h = int(figure_w), int(figure_h)
    photo = tk.PhotoImage(master=canvas, width=figure_w, height=figure_h)

    # Position: convert from top-left anchor to center anchor
    canvas.create_image(loc[0] + figure_w/2, loc[1] + figure_h/2, image=photo)

    # Unfortunately, there's no accessor for the pointer to the native renderer
    tkagg.blit(photo, figure_canvas_agg.get_renderer()._renderer, colormode=2)

    # Return a handle which contains a reference to the photo object
    # which must be kept live or else the picture disappears
    return photo"""

if __name__ == "__main__":
	data_path = "D:\\Etiology\\flipped.csv"
	if exists(data_path):
		df = pd.read_csv(data_path)
		df.index = df.index.map(str)
	else:
		df = pd.DataFrame(columns=["Flipped"])

	for fn in glob.glob(r"D:\Etiology\screenshots\*\*"):
		accnum = basename(fn)
		accnum = accnum[:accnum.find('_')]
		if accnum in df.index:
			continue
		df.loc[accnum, "Flipped"] = easygui.boolbox(msg='Flipped?', image=fn)

	df.to_csv(data_path)

"""w, h = 250, 550
window = tk.Tk()
window.title(basename(fn))
canvas = tk.Canvas(window, width=w, height=h)
canvas.pack()

# Create the figure we desire to add to an existing canvas
fig = mpl.figure.Figure(figsize=(2, 5))
ax = fig.add_axes([0, 0, 1, 1])

I=plt.imread(fn)
#I = rescale(I, 5, mode='constant')
ax.imshow(I, cmap='gray')
ax.axis('off')

fig_x, fig_y = 0, 0
fig_photo = draw_figure(canvas, fig, loc=(fig_x, fig_y))
fig_w, fig_h = fig_photo.width(), fig_photo.height()
#canvas.create_line(200, 50, fig_x + fig_w / 2, fig_y + fig_h / 2)
#canvas.create_text(200, 50, text="Zero-crossing", anchor="s")

# Let Tk take over
tk.mainloop()

if input() == 'q':
	break"""