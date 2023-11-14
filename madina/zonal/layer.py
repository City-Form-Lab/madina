from zonal import Zonal


class Layers:
    """
    Represents a collection of Zonal layers.
    """

    def __init__(self, layers=None):
        self.layers = []
        self.label_layers = {}
        self._idx = -1

        if layers:
            for layer in layers:
                self.layers.append(layer.label)
                self.label_layers[layer.label] = layer
        return

    def __getitem__(self, item):
        if type(item) == str:
            if item in self.label_layers:
                return self.label_layers[item]

            raise KeyError(f"No layer in the Zonal object has the label {item}")

        elif type(item) == int:
            label = self.layers[item]
            return self.label_layers[label]

        else:
            raise KeyError("Only indices of type string and int are allowed.")

    def __setitem__(self, key, value):
        """
        If `key` is a string, layer with label `key` will be replaced with `value`.
        If `key` is an integer, the layer at pos `key` will be replaced.

        Returns:
            None

        Raises:
            KeyError if integer `key` is out of bounds, or no layer exists in Zonal with string `key`.
        """

        if type(key) == str:
            if key not in self.label_layers:
                raise KeyError(f"Key {key} does not exist in Zonal. Remember: only existing layers can be re-set by label")
            self.label_layers[key] = value
            return
        elif type(key) == int:
            if key >= len(self.layers):
                raise IndexError(f"Index {key} is out of bounds. To add a new layer use `add`.")
            replaced_layer = self.layers[key]
            del self.label_layers[replaced_layer]
            self.layers[key] = value
            self.label_layers[value.label] = value
        else:
            raise TypeError("Only keys with type str or int may be used to set.")

    def __str__(self):
        """
        Displays the order of the `Zonal` layers.

        Returns:
            A string of layer position: label names

        """
        output = ""
        reverse = self.layers[::-1]

        for i in range(len(reverse)):
            output += f"\nPosition {len(reverse) - i} ----- {reverse[i]}"

        return output

    def __contains__(self, item):
        if type(item) == str:
            return item in self.label_layers
        elif isinstance(item, Layer):
            return item == self.label_layers[item.label]
        else:
            raise TypeError("item must be an instance of the `Layer` class or a string label")

    def __iter__(self):
        return self

    def __next__(self):
        self._idx += 1
        if self._idx >= len(self.layers):
            self._idx = -1
            raise StopIteration
        else:
            return self.layers[self._idx]

    def add(self, layer, pos=None, first=False, before: str = None, after=None):
        """
        Adds a new layer `layer` to the top of the stack.
        If `pos` is specified, ignores the `before`, `after` and `last` parameters.
        """

        if pos:
            if type(pos) != int:
                raise TypeError(f"`pos` must have an int type")
            return self.insert_at(pos, layer)

        if first:
            return self.insert_at(0, layer)

        if before:
            pos = self.layers.index(before)
            return self.insert_at(pos, layer)

        if after:
            pos = self.layers.index(after)
            pos += 1
            return self.insert_at(pos, layer)

        self.insert_at(len(self.layers), layer)
        return

    def insert_at(self, pos: int, layer):
        """
        Inserts new layer `layer` into the specified position `pos`.

        Returns:
            None

        Raises:
            KeyError if a layer with the same label is already in the Zonal
        """
        if type(pos) != int:
            raise ValueError(f"{pos} must have an int type")

        if layer.label in self.label_layers:
            raise KeyError(f"Layer with label {layer.label} is already in Zonal object")

        label = layer.label

        self.layers.insert(pos, label)
        self.label_layers[label] = layer

        return

    def get_layer_at_pos(self, pos: int):
        if pos not in range(len(self.layers)):
            raise IndexError(f"Chosen index {pos} out of bounds")

        label = self.layers[pos]

        return self.label_layers[label]

    def pos_at_label(self, label):
        return self.layers.index(label)


class Layer:
    """
    Represents a `Zonal` layer with name `label`, source `file_path` and crs `original_crs`.
    """

    def __init__(self, label: str, gdf, show: bool, original_crs: str, file_path: str, default_colors = None, **kwargs):
        self.gdf = gdf
        self.label = label
        self.show = show
        self.crs = original_crs
        self.file_path = file_path
        self.default_colors = default_colors

        self.other_fields = {} | kwargs

        return

    def set_style(self, params):
        color, color_by_attribute, color_method = None, None, "single_color"
        if 'color' in params:
            color = params['color']

        if 'color_by_attribute' in params:
            color_by_attribute = params['color_by_attribute']

        if 'color_method' in params:
            color_method = params['color_method']

        self.color_layer(self.label, color_by_attribute, color_method, color)
        return

    def color_layer(self, layer_name, color_by_attribute=None, color_method="single_color", color=None):
        if layer_name in self.default_colors.keys() and color_by_attribute is None and color is None:
            # set default colors first. all default layers call without specifying "color_by_attribute"
            # default layer creation always calls self.color_layer(layer_name) without any other parameters
            color = self.default_colors[layer_name].copy()
            color_method = "single_color"
            if type(color) is dict:
                # the default color is categorical..
                color_by_attribute = color["__attribute_name__"]
                color_method = "categorical"
        Zonal.color_gdf(
            self.gdf,
            color_by_attribute=color_by_attribute,
            color_method=color_method,
            color=color
        )
        return








