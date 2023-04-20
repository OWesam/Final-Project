//neural network class

pub mod matrix;


extern crate rand;

pub struct hidden_layer {
    // layers;
    weights : Matrix,
    biases : Matrix,
    output : Matrix,
}

impl hidden_layer {
    pub fn init(Self, num_inputs: u32, num_neurons: u32)  {

    }
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

    // pub fn calculate_fitness() -> u64 {
    //     let fitness = 1000 * num_food + num_moves;
    //     fitness
    // }
}
