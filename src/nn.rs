                    /* --------------------  neural network Struct and traits  ---------------------*/

// Notes :
/* 
    Inputs: 
        if there is a body part or wall on the left side of the snake’s head (0 or 1 value) 
        if there is body part or wall in front of the snake (0 or value)
        if there is a body part or wall on the right side of the snake’s head (0 or 1 value) 
        food position relative to the snake
    
    For simplicity: one hidden layer with 6 neurons

    Outputs: (outputs are based on snake's current direction)
        Turn left
        Turn Right
        go forward (do nothing)

*/

use std::thread;

use Super::matrix::Matrix;
pub mod matrix;

extern crate rand;

pub struct NN<'a> {
    // layers;
    layers : Vec<usize>,
    weights : Vec<Matrix>,
    biases : Vec<Matrix>,
    data : Vec<Matrix>,
    activation: Activation<'a>, 
    learning_rate,
}

impl NN<'_> {
    pub fn new<'a>(layers: Vec<usize>, activation: Activation<'a>, learning_rate: f64) -> NN {
    let mut biases = vec![];
    let mut weights = vec![];

    for i in 0..layers.len() - 1 {
        weights.push(Matrix::random(layers[i + 1], layers[i]));
        biases.push(Matrix::random(layers[i + 1], 1));
    }

        NN { 
            layers: Vec::new(),
            weights,
            biases,
            data: Vec![],
            activation,
            learning_rate,
        }
    }


    // move forward in neural network
    pub fn feed_forwards(&self, inputs: Vec<f64>) -> Vec<f64> {
        if inputs.length != self.layers[0] {
            panic!("Input length differnet than the input number of neurons");
        }
    
        let mut current = Matrix::from(vec![inputs].transpose()); 
        self.data = vec![current.clone()];
        thread::spawn(|| {
            for i in 0..self.layers.length - 1 {
                currrent = self.weights[i]
                    .multiply(&current)
                    .add(&self.biases[i])
                    .map(self.activation.function);
                self.data.push(current.clone());
            }
        });
            
        // to_owned -> basically clones data
        current.data[0].to_owned()

    }

    // move to backwards in network  
    pub fn back_propagate(&self, outputs: Vec<f64>, targets: Vec<f64>)   { // -- compare to correct output and provide feedback
        if target.len() != self.layers[self.layers.len() - 1] {
            panic!("target length different than output length");
        }

        // get errors after comparing to correct output
        let mut parsed = Matrix::from(vec![outputs]);
        let mut errors = Matrix::from(vec![targets]).subtract(parsed);

        // undo activation function
        let mut gradients = parsed.map(self.activation.derivative);

        for i in (0..self.layers.len() - 1).rev() {
            gradients = gradients.dot_multiply(&errors).map(&|x| x * self.learning_rate);
            self.weights[i] = self.weights.add(gradients.multiply(&self.data[i].transpose()));
            self.biases[i] = self.biases.add(gradients);

        }
    }

    pub fn learn() {

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
