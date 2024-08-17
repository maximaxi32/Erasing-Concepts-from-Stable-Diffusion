#Much of this code is borrowed
import matplotlib.pyplot as plt
import textwrap
from PIL import Image

def image_grid(images, outpath=None, column_titles=None, row_titles=None):

    n_rows = len(images)
    n_cols = len(images[0])

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols,
                            figsize=(n_cols, n_rows), squeeze=False)

    for row, _images in enumerate(images):

        for column, image in enumerate(_images):
            ax = axs[row][column]
            ax.imshow(image)
            if column_titles and row == 0:
                ax.set_title(textwrap.fill(
                    column_titles[column], width=12), fontsize='x-small')
            if row_titles and column == 0:
                ax.set_ylabel(row_titles[row], rotation=0, fontsize='x-small', labelpad=1.6 * len(row_titles[row]))
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.subplots_adjust(wspace=0, hspace=0)

    if outpath is not None:
        plt.savefig(outpath, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.tight_layout(pad=0)
        image = figure_to_image(plt.gcf())
        plt.close()
        return image
    
def figure_to_image(figure):

    figure.set_dpi(300)

    figure.canvas.draw()

    return Image.frombytes('RGB', figure.canvas.get_width_height(), figure.canvas.tostring_rgb())
