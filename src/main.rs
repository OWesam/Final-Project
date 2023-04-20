mod constants;
mod game;
mod brain;

mod render;

extern crate rayon;

use rayon::prelude::*;

use crate::constants::*;
use crate::game::{Game};
use crate::brain::{Brain};

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
    render.run_random();
}