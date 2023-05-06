                    /* --------------------  neural network Struct and traits  ---------------------*/

// Notes :
/* 
    Inputs: 
        if there is a body part or wall on the left side of the snake’s head (0 or 1 value) 
        if there is body part or wall in front of the snake (0 or 1 value)
        if there is a body part or wall on the right side of the snake’s head (0 or 1 value) 
        food position relative to the snake
        
    For simplicity: one hidden layer with 6 neurons

    Outputs: (outputs are based on snake's current direction)
        Turn left
        Turn Right
        go forward (do nothing)

*/

use std::thread;

use crate::matrix::Matrix;
use crate::activation::Activation;


extern crate rand;

pub struct NN<'a> {
    // layers;
    layers : Vec<usize>,
    weights : Vec<Matrix>,
    biases : Vec<Matrix>,
    data : Vec<Matrix>,
    activation: Activation<'a>, 
    learning_rate: f64,
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
            data: vec![],
            activation,
            learning_rate,
        }
    }


    // move forward in neural network
    pub fn feed_forwards(&self, inputs: Vec<f64>) -> Vec<f64> {
        if inputs.len() != self.layers[0] {
            panic!("Input length differnet than the input number of neurons");
        }
    
        let mut current = Matrix::from(vec![inputs]).transpose(); 
        self.data = vec![current.clone()];
        
        for i in 0..self.layers.len() - 1 {
            current = self.weights[i]
                .multiply(&current)
                .add(&self.biases[i])
                .map(self.activation.function);
            self.data.push(current.clone());
        }
        
            
        // to_owned -> basically clones data
        current.data[0].to_owned()

    }

    // move to backwards in network  
    pub fn back_propagate(&self, outputs: Vec<f64>, targets: Vec<f64>)   { // -- compare to correct output and provide feedback
        if targets.len() != self.layers[self.layers.len() - 1] {
            panic!("target length different than output length");
        }

        // get errors after comparing to correct output
        let mut parsed = Matrix::from(vec![outputs ]);
        let mut errors = Matrix::from(vec![targets ]).subtract(&parsed);

        // undo activation function
        let mut gradients = parsed.map(self.activation.derivative);

        for i in (0..self.layers.len() - 1).rev() {
            gradients = gradients.dot_multiply(&errors).map(&|x| x * self.learning_rate);
            self.weights[i] = self.weights[i].add(&gradients.multiply(&self.data[i].transpose()));
            self.biases[i] = self.biases[i].add(&gradients);

            errors = self.weights[i].transpose().multiply(&errors);
			gradients = self.data[i].map(self.activation.derivative);
        }
    }

    // pub fn learn(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u16) {
    //     for i in 1..=epochs {
	// 		if epochs < 100 || i % (epochs / 100) == 0 {
	// 			println!("Epoch {} of {}", i, epochs);
	// 		}
	// 		for j in 0..inputs.len() {
	// 			let outputs = self.feed_forward(inputs[j].clone());
	// 			self.back_propogate(outputs, targets[j].clone());
	// 		}
	// 	}
    // }

   
}

pub struct Trainer<'a> {
    model: NN<'a>,
    lr: f64,
    gamma: f64,
}

impl Trainer<'_> {
    pub fn new(model: NN, lr: f64, gamma: f64) -> Trainer {
        Trainer {
            model,
            lr,
            gamma,
        }
    }
    pub fn train_step(&self, states: Vec<Vec<f64>>, actions: Vec<Vec<f64>>, rewards: Vec<i32>, next_state: Vec<Vec<f64>>, done: Vec<bool>) {
        
        let prediction = self.model.feed_forwards(states[states.len()-1]);
        let target = prediction.clone();
        for i in 0..states.len() {
            
            let q_new = rewards[i];
            
            if !done[i] {
                q_new = (rewards[i] as f64 + self.gamma * ((self.model.feed_forwards(next_state[i]).iter().max_by(|a, b| a.partial_cmp(b).unwrap()))).unwrap()) as i32;
            }
            //let _ = values.into_iter().min_by(|a, b| a.partial_cmp(b).unwrap());
            let max = actions[i][0];
            let max_index = 0;
            for j in 0..actions[i].len() {
                if(actions[i][j] > max) {
                    max = actions[i][j];
                    max_index = j;
                }
            }
            //let f64_index: u64 =  ((actions[i].into_iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap())) as u64 ;
            
            target[max_index] = q_new.into(); //figure this out
        }
        self.model.back_propagate(prediction, target);
    }
    
}