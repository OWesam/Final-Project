extern crate rand;

use rand::Rng;
use rand::distributions::{Distribution, Uniform};

pub struct Brain {
    
}
impl Brain {
    pub fn random() {
        let mut rng = rand::thread_rng();
        let die = Uniform::from(1..7);
        let throw = die.sample(&mut rng);
        println!("Roll the die: {}", throw);
        //println!("{}",rng.gen_range(0..2));
        

    }
}

