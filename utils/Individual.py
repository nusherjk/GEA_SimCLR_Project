import uuid

class Individual:
    def __init__(self, layer_dims, activations):
        self.id = uuid.uuid4().hex
        self.architecture = [
            {'in': layer_dims[i], 'out': layer_dims[i+1], 'activation': activations[i]}
            for i in range(len(layer_dims) - 1)
        ]
        self.fitness = None
