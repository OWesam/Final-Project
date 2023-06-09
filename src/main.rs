mod constants;
mod game;
mod brain;
mod render;

pub mod nn;
pub mod matrix;
pub mod activation;

extern crate rayon;


use rayon::prelude::*;

use crate::brain::Brain;
use crate::constants::*;
use crate::game::{Game};

use crate::render::Render;

enum GameType {
    Human,
    GeneticAlgorithm,
    QLearning,
}

fn main() {
    
    render_game();
}





// --------------------------------------------------------------------------------------
// ----------------------------------Neural Network--------------------------------------
// --------------------------------------------------------------------------------------


// --------------------------------------------------------------------------------------
// ----------------------------------Human Game------------------------------------------
// --------------------------------------------------------------------------------------

fn render_game() {
    let mut render = Render::new();
    //render.run();
    render.run_network(&vec![0,1,3]);
}