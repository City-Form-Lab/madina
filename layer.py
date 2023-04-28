class Layers:
    """
    Represents a collection of Zonal layers.
    """

    def __init__(self, layers=None):
        self.layers = []
        self.label_layers = {}

        if layers:
            for layer in layers:
                self.layers.append(layer.label)
                self.label_layers[layer.label] = layer

            print(self)

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

        print(self)
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

    def __init__(self, label: str, gdf, show: bool, original_crs: str):
        self.gdf = gdf
        self.label = label
        self.show = show
        self.original_crs = original_crs

        self.other_fields = {}

        return

    

