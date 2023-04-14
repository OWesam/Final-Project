


pub struct network {
    // layers;

}

impl NN {
    pub fn new() -> NN {
        NN { layers: Vec::new() }
    }

    fn mutate(&mut self) {
        for layer in &mut self.layers {
            layer.mutate();
        }
    }

    pub fn add(&mut self, layer: Layer) -> bool {
        if self.layers.is_empty() || self.layers.last().unwrap().num_neurons == layer.num_inputs {
            self.layers.push(layer);
            true
        } else {
            false
        }
    }
}
