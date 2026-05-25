import numpy as np
import matplotlib.pyplot as plt


def _box_stats(data):
    data = np.asarray(data)

    q1 = np.percentile(data, 25)
    med = np.percentile(data, 50)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    low = data[data >= q1 - 1.5 * iqr].min()
    high = data[data <= q3 + 1.5 * iqr].max()

    outliers = data[(data < low) | (data > high)]

    return q1, med, q3, low, high, outliers


def boxplot2d(
    df,
    x,
    y,
    ax,
    box_edgecolor='blue',
    median_color='green',
    whisker_color='black',
    cap_size=0.1
):
    """
    Draw a true 2D boxplot on a given Axes.
    """

    x_data = df[x].values
    y_data = df[y].values

    # statistics
    x_q1, x_med, x_q3, x_min, x_max, x_out = _box_stats(x_data)
    y_q1, y_med, y_q3, y_min, y_max, y_out = _box_stats(y_data)

    # --- central 2D box (single rectangle) ---
    ax.add_patch(
        plt.Rectangle(
            (x_q1, y_q1),
            x_q3 - x_q1,
            y_q3 - y_q1,
            fill=False,
            edgecolor=box_edgecolor,
            linewidth=1.5
        )
    )

    # --- median lines ---
    ax.plot([x_med, x_med], [y_q1, y_q3], color=median_color, linewidth=2)
    ax.plot([x_q1, x_q3], [y_med, y_med], color=median_color, linewidth=2)

    # --- whiskers ---
    ax.plot([x_min, x_q1], [y_med, y_med], color=whisker_color)
    ax.plot([x_q3, x_max], [y_med, y_med], color=whisker_color)

    ax.plot([x_med, x_med], [y_min, y_q1], color=whisker_color)
    ax.plot([x_med, x_med], [y_q3, y_max], color=whisker_color)

    # --- caps (min/max lines) ---
    ax.plot([x_min, x_min], [y_med - cap_size,
            y_med + cap_size], color=whisker_color)
    ax.plot([x_max, x_max], [y_med - cap_size,
            y_med + cap_size], color=whisker_color)

    ax.plot([x_med - cap_size, x_med + cap_size],
            [y_min, y_min], color=whisker_color)
    ax.plot([x_med - cap_size, x_med + cap_size],
            [y_max, y_max], color=whisker_color)

    # --- outliers ---
    ax.scatter(x_out, np.full_like(x_out, y_med),
               facecolors='none', edgecolors=whisker_color)

    ax.scatter(np.full_like(y_out, x_med), y_out,
               facecolors='none', edgecolors=whisker_color)

    ax.set_xlabel(x)
    ax.set_ylabel(y)

    return ax
