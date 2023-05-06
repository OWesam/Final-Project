// Learning algorithm class

// use nn::NN;
use super::{nn::NN, nn::Trainer};

use crate::Render;

use crate::activation::SIGMOID;

use crate::game::{Block, Direction, Game};
use std::collections::VecDeque;
extern crate rand;

use rand::Rng;

pub const MAX_MEMORY : usize = 100000;
pub const BATCH_SIZE : usize = 1000;
pub const LR : f64 = 0.001;


pub struct Brain<'a> {
    num_games: f64,
    epsilon: f64,
    gamma: f64,
    memory: VecDeque<(Vec<f64>, Vec<f64>, i32,Vec<f64>, bool)>,
    model: NN<'a>,
    trainer: Trainer<'a>,
    render: Render,
}


impl Brain<'_> {
    pub fn new(&self) -> Brain {
        Brain {
            num_games: 0.0,
            epsilon: 0.0,
            gamma: 0.9,
            memory: VecDeque::with_capacity(MAX_MEMORY),
            model: NN::new(vec![11,256,3], SIGMOID, LR),
            trainer: Trainer::new(self.model, LR, self.gamma),
            render: Render::new(),
        }
    }

    pub fn get_state(self, game: &Game) -> Vec<f64> {
        let head = game.snake.body[0];
        let point_r = head.position;
        point_r.offset(1, 0);
        let point_u = head.position;
        point_u.offset(0, -1);
        let point_l = head.position;
        point_l.offset(-1, 0);
        let point_d = head.position;
        point_d.offset(0, 1);

        let dir_l = game.snake.direction == Direction::LEFT;
        let dir_r = game.snake.direction == Direction::RIGHT;
        let dir_u = game.snake.direction == Direction::UP;
        let dir_d = game.snake.direction == Direction::DOWN;

        let state = vec![((dir_r && (game.snake.check_collide_body(point_r) || game.snake.check_collide_wall(point_r)))
                        || (dir_l && (game.snake.check_collide_body(point_l) || game.snake.check_collide_wall(point_l)))
                        || (dir_u && (game.snake.check_collide_body(point_u) || game.snake.check_collide_wall(point_u)))
                        || (dir_d && (game.snake.check_collide_body(point_d) || game.snake.check_collide_wall(point_d)))) as i32 as f64,

                        ((dir_u && (game.snake.check_collide_body(point_r) || game.snake.check_collide_wall(point_r))) 
                        || (dir_d && (game.snake.check_collide_body(point_l) || game.snake.check_collide_wall(point_l)))
                        || (dir_l && (game.snake.check_collide_body(point_u) || game.snake.check_collide_wall(point_u)))
                        || (dir_r && (game.snake.check_collide_body(point_d) || game.snake.check_collide_wall(point_d)))) as i32 as f64,

                        ((dir_d && (game.snake.check_collide_body(point_r) || game.snake.check_collide_wall(point_r)))
                        || (dir_u && (game.snake.check_collide_body(point_l) || game.snake.check_collide_wall(point_l)))
                        || (dir_r && (game.snake.check_collide_body(point_u) || game.snake.check_collide_wall(point_u)))
                        || (dir_l && (game.snake.check_collide_body(point_d) || game.snake.check_collide_wall(point_d)))) as i32 as f64,

                        dir_l as i32 as f64,
                        dir_r as i32 as f64,
                        dir_u as i32 as f64,
                        dir_d as i32 as f64,

                        (game.get_food_pos().x < head.position.x) as i32 as f64,
                        (game.get_food_pos().x > head.position.x) as i32  as f64,
                        (game.get_food_pos().y < head.position.y) as i32 as f64,
                        (game.get_food_pos().y > head.position.y) as i32  as f64
        ];

        state
    }

    pub fn remember(self, state: Vec<f64>, action: Vec<f64>, reward: i32, next_state: Vec<f64>, done: bool)   {
        let (x, y, z, f, g) = (state, action, reward, next_state, done);
        self.memory.push_back((x,y,z,f,g));
    }

    pub fn train_long_memory(self)  {
        let mini_sample = self.memory;
        if self.memory.len() > BATCH_SIZE {
            let mut rng = rand::thread_rng();
            mini_sample = VecDeque::new();
            mini_sample.push_back(*self.memory.get(rng.gen_range(0, self.memory.len())).unwrap()); 
        }        

        let states = Vec::new();
        let actions = Vec::new();
        let rewards = Vec::new();
        let next_state = Vec::new();
        let done = Vec::new();

        for i in 0..mini_sample.len() {
            states.push(mini_sample[i].0);
            actions.push(mini_sample[i].1);
            rewards.push(mini_sample[i].2);
            next_state.push(mini_sample[i].3);
            done.push(mini_sample[i].4);
        }

        self.trainer.train_step(states, actions, rewards, next_state, done);
    }

    pub fn train_short_memory(&self,states: Vec<Vec<f64>>, actions: Vec<Vec<f64>>, rewards: Vec<i32>, next_state: Vec<Vec<f64>>, done: Vec<bool>)  {
        self.trainer.train_step(states, actions, rewards, next_state, done);
    }

    pub fn get_action(self, state: Vec<f64>) -> Vec<f64> {
        self.epsilon = 80.0 - self.num_games;
        let final_move = vec![0.0,0.0,0.0];
        let mut rng = rand::thread_rng();
        let random = rng.gen_range(0, 200);
        let rand_move = 0;
        if (random as f64) < self.epsilon {
            rand_move = rng.gen_range(0,2);
            final_move[rand_move] = 1.0;
        } else {
            let prediction = self.model.feed_forwards(state);
            let temp = prediction.iter().max_by(|a, b| a.partial_cmp(b).unwrap());
            rand_move = match temp {
                None => panic!(""),
                Some(i) => *i as usize
            };
            final_move[rand_move] = 1.0;
        }
        
        final_move
    }

    pub fn train(&self) {
        let mut scores : Vec<f64> = Vec::new();
        let mut total_score = 0;
        let mut record = 0;
        let mut brain = Brain::<'_>::new(&self);
        let mut game = Game::new();
        let done = Vec::new();
        let states_old = Vec::new();
        let final_moves = Vec::new();
        let rewards = Vec::new();
        let states_new = Vec::new();
        let mut i = 0;

        while true {
            let state_old = brain.get_state(&game);
            states_old.push(state_old);
            let final_move = brain.get_action(state_old);
            final_moves.push(final_move);
            let (reward,still_playing) = game.snake.perform_next(&mut game.get_food_pos());
            rewards.push(reward);

            let final_move_usize = Vec::new(); //final_move;
            for i in 0..final_move.len() {
                final_move_usize.push( final_move[i] as usize); 
            }

            self.render.handle_network_events(& mut game, &final_move_usize);
            let state_new = brain.get_state(&game);
            states_new.push(state_new);
            done.push(still_playing);

            self.train_short_memory(states_old, final_moves, rewards, states_new, done);
            brain.remember(state_old, final_move, reward.try_into().unwrap(), state_new, done[i]);

            if !game.snake.alive {
                game.reset();
                brain.train_long_memory();
                brain.num_games = brain.num_games + 1.0;
                println!("game,{}", brain.num_games);
            }
            i += 1;
        }
    }

}