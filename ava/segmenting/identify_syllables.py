"""
Identify syllables.

"""
__date__ = "December 2023"


from itertools import repeat
from joblib import Parallel, delayed
import matplotlib
from matplotlib.path import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
plt.switch_backend('agg')
import numpy as np
try: # Numba >= 0.52
	from numba.core.errors import NumbaPerformanceWarning
except ModuleNotFoundError:
	try: # Numba <= 0.45
		from numba.errors import NumbaPerformanceWarning
	except (NameError, ModuleNotFoundError):
		pass
import os
from scipy.spatial import ConvexHull
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning
import umap
import warnings

from ava.plotting.tooltip_plot import tooltip_plot
from ava.segmenting.utils import get_spec, get_audio_seg_filenames, \
		_read_onsets_offsets


CMAP = ["red", "orange", "yellow", "green", "blue"]

def identify_syllables_post_vae(dc, out_syll_fname, verbose=True, 				
								num_imgs=2000, tooltip_output_dir='temp', make_tooltip=True, img_fn='temp.pdf', cmap=CMAP):
	"""
	Manually identify syllables by selecting regions of UMAP latent mean projection.

	First, a tooltip plot of the spectrogram latent means will be made (using
	`ava.plotting.tooltip_plot`) and saved to `tooltip_output_dir`. You should
	open this plot and see which regions of the UMAP contain syllables. Then, when prompted, press return to identify syllables, Then enter the syllables of interest, and the vertices of a polygon (x, y) in the UMAP projection containing a syllable, following the prompts. You will be able to see the selected syllable regions in the image save at `img_fn`, by default `'temp.pdf'`. When you are finished identifying syllable regions, press `'q'`. A list of syllables, e.g., 'A', 'B', is output, corresponding to each sample.
	Doesn't support datasets that are too large to fit in memory.

	Parameters
	----------
	dc : ava.data.data_container.DataContainer
		DataContainer object
	out_syll_fname : str
		Filename for the text file containing syllable labels.
	verbose : bool, optional
		Defaults to ``True``.
	num_imgs : int, optional
		Number of images for tooltip plot. Defaults to ``2000``.
	tooltip_output_dir : str, optional
		Where to save tooltip plot. Defaults to ``'temp'``.
	make_tooltip : bool, optional
		Defaults to ``True``.
	img_fn : str, optional
		Where to save
	cmap : list, optional
		Colormap for coloring points containing syllables.
	"""
	# Get UMAP embedding.
	embed = dc.request('latent_mean_umap')
	labels = [''] * len(embed)
	colors = ['k'] * len(embed)
	syllable_num = 0
	first_iteration = True
	# Keep drawing boxes around noise.
	while True:
		_plot_helper(embed, colors, filename=img_fn, verbose=verbose)
		if first_iteration and make_tooltip:
			if verbose:
				print("Writing html plot:")
			first_iteration = False
			title = "Identify syllables:"
			specs = dc.request('specs')
			tooltip_plot(embed, specs, num_imgs=num_imgs, title=title, \
					output_dir=tooltip_output_dir, grid=True)
			if verbose:
				print("\tDone.")
		if input("Press [q] to quit identifying syllables or \
				[return] to continue: ") == 'q':
			break
		print("Enter the syllable name: ")
		syllable = _get_input("syllable: ", syllable=True)
		mask, labels = _make_convex_hull(embed, syllable, labels)
		with open(out_syll_fname, "w") as output:
			output.write(str(labels))
		# Update scatter colors.
		colors = _update_colors(colors, cmap, mask, syllable_num)
		syllable_num += 1
	return labels


def _plot_helper(embed, colors, title="", filename='temp.pdf', verbose=True):
	"""Helper function to plot a UMAP projection with grids."""
	plt.scatter(embed[:,0], embed[:,1], c=colors, s=0.9, alpha=0.7)
	delta = 1
	if np.max(embed) - np.min(embed) > 20:
		delta = 5
	min_xval = int(np.floor(np.min(embed[:,0])))
	if min_xval % delta != 0:
		min_xval -= min_xval % delta
	max_xval = int(np.ceil(np.max(embed[:,0])))
	if max_xval % delta != 0:
		max_xval -= (max_xval % delta) - delta
	min_yval = int(np.floor(np.min(embed[:,1])))
	if min_yval % delta != 0:
		min_yval -= min_yval % delta
	max_yval = int(np.ceil(np.max(embed[:,1])))
	if max_yval % delta != 0:
		max_yval -= (max_yval % delta) - delta
	for x_val in range(min_xval, max_xval+1):
		plt.axvline(x=x_val, lw=0.5, alpha=0.7)
	for y_val in range(min_yval, max_yval+1):
		plt.axhline(y=y_val, lw=0.5, alpha=0.7)
	plt.title(title)
	plt.savefig(filename)
	plt.close('all')
	if verbose:
		print("Grid plot saved to:", filename)


def _get_input(query_str, syllable=False):
	"""Get float-valued input."""
	while True:
		if syllable:
			temp = str(input(query_str))
			return temp
		try:
			temp = float(input(query_str))
			return temp
		except:
			print("Unrecognized input!")
			pass


def _update_colors(colors, cmap, mask, syllable_num):
	"""Color if embed is polygon, black otherwise."""
	for i in range(len(colors)):
		if colors[i] == 'k' and mask[i]:
			colors[i] = cmap[syllable_num]
	return colors


def _convex_hull_vertices(x, y):
    points = np.vstack((x, y)).T
    hull = ConvexHull(points)
    vertices = np.array([points[i] for i in hull.vertices])
    return vertices


def _points_inside_convex_hull(vertices, x, y):
	points = np.vstack((x, y)).T
	p = Path(vertices, closed=True)
	mask = p.contains_points(points)
	return mask


def _make_convex_hull(embed, syllable, labels):
	xs = []
	ys = []
	while True:
		x = _get_input("x: ")
		y = _get_input("y: ")
		xs.append(x)
		ys.append(y)
		if len(xs) and len(ys) > 3:
			convex_hull_vertices = _convex_hull_vertices(xs, ys)
			mask = _points_inside_convex_hull(convex_hull_vertices, embed[:,0], embed[:,1])
			if input("Press [q] to quit identifying syllable " + str(syllable) + " or [return] to continue: ") == 'q':
				break
	labels = np.where(~mask, labels, syllable)
	return mask, labels


if __name__ == '__main__':
	pass


###
