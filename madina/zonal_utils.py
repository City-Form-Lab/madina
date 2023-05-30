import random
import numpy as np


def get_color_column(method, gdf, by_attribute, color_scheme):
    if method == "categorical":
        if not color_scheme:
            color_scheme = {"__other__": [255, 255, 255]}

            for distinct_value in gdf[by_attribute].unique():
                color_scheme[distinct_value] = [random.random() * 255, random.random() * 255, random.random() * 255]

        column = []
        for value in gdf[by_attribute]:
            if value in color_scheme.keys():
                column.append(color_scheme[value])
            else:
                column.append(color_scheme["__other__"])

    elif method == "single":
        color_scheme = [random.random() * 255, random.random() * 255, random.random() * 255]
        column = [color_scheme] * len(gdf)

    elif method == "gradient":
        cbc = gdf[by_attribute]  # color by column
        nc = 255 * (cbc - cbc.min()) / (cbc.max() - cbc.min())  # normalized column
        column = [[255 - v, 0 + v, 0] if not np.isnan(v) else [255, 255, 255] for v in
                  list(nc)]  # convert normalized values to color spectrum.

    elif method == 'quantile':
        scaled_percentile_rank = 255 * gdf[by_attribute].rank(pct=True)
        column = [[255.0 - v, 0.0 + v, 0] if not np.isnan(v) else [255, 255, 255] for v in
                  scaled_percentile_rank]  # convert normalized values to color spectrum.

    else:
        raise ValueError("Method must be one of 'single', 'categorical', or 'gradient'")

    return column
