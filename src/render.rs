extern crate glutin_window;
extern crate graphics;
extern crate opengl_graphics;
extern crate piston;

use crate::constants::*;
use crate::game::{Block, Direction, Game};

use glutin_window::GlutinWindow;
use opengl_graphics::{GlGraphics, OpenGL};
use piston::event_loop::{EventLoop, EventSettings, Events};
use piston::input::keyboard::Key;
use piston::input::{Button, PressEvent, RenderArgs, RenderEvent, UpdateEvent}; //UpdateArgs
use piston::window::WindowSettings;

pub struct Render {
    window: GlutinWindow,
    events: Events,
    gl: GlGraphics,
    // maybe frame_iteration here
}

impl Render {
    pub fn new() -> Render {
        Render {
            window: WindowSettings::new(
                NAME,
                [BOARD_WIDTH as u32 * BLOCK_SIZE * 3, BOARD_HEIGHT as u32 * BLOCK_SIZE * 3], 
            )
            .graphics_api(OpenGL::V3_2)
            .vsync(true)
            .exit_on_esc(true)
            .build()
            .unwrap(),
            events: Events::new(EventSettings::new().ups(RENDER_UPS).max_fps(RENDER_FPS_MAX)),
            gl: GlGraphics::new(OpenGL::V3_2),
        }
    }

    // pub fn run(&mut self) {
    //     let mut game = Game::new();
    //     game.init();

    //     while let Some(e) = self.events.next(&mut self.window) {
    //         if let Some(args) = e.render_args() {
    //             self.render_game(&args, &game);
    //         }

    //         if let Some(args) = e.update_args() {
    //             game.next_tick(args.dt);
    //         }

    //         if let Some(button) = e.press_args() {
    //             self.handle_events(button, &mut game);
    //         }
    //     }
    // }

    pub fn run_network(&mut self, action: &Vec<usize>)   {
        let mut game = Game::new();
        game.init();

        while let Some(e) = self.events.next(&mut self.window) {
            if let Some(args) = e.render_args() {
                self.render_game(&args, &game);
            }

            if let Some(args) = e.update_args() {
                game.next_tick(args.dt);
            }

            if let Some(button) = e.press_args() {
                let act = self.handle_network_events(&mut game, action);
            }
        }
    }
    
    // fn handle_events(&mut self, button: Button, game: &mut Game) {
    //     match button {
    //         Button::Keyboard(key) => match key {
    //             Key::Up => game.update(Direction::UP),
    //             Key::Down => game.update(Direction::DOWN),
    //             Key::Left => game.update(Direction::LEFT),
    //             Key::Right => game.update(Direction::RIGHT),
    //             Key::Space => game.init(),
    //             _ => {}
    //         },
    //         _ => {}
    //     }
    // }
    
    pub fn handle_network_events(&mut self, game: &mut Game, action: &Vec<usize>) { 

        let mut clock_wise = vec![Direction::RIGHT, Direction::DOWN, Direction::LEFT, Direction::UP];
        let index = clock_wise.iter().position(|&r| r == game.snake.direction).unwrap();

        let one_zero_zero = vec![1,0,0];
        let zero_one_zero = vec![0,1,0];
        let zero_zero_one = vec![0,0,1];

        let mut next_index = 0;

        match action {
            one_zero_zero => game.update(clock_wise[index]),
            zero_one_zero => {
                next_index = (index + 1) % 4;
                game.update(clock_wise[next_index]);
            },
            zero_zero_one => {
                next_index = (index - 1) % 4;
                game.update(clock_wise[next_index]);
            }
            _ => {}
        };
    }

    fn render_game(&mut self, args: &RenderArgs, game: &Game) {
        self.gl.draw(args.viewport(), |_c, g| {
            graphics::clear(BLACK, g);
        });
        for b in game.snake.body.iter() {
            self.render_block(&b);
        }
        self.render_block(&game.food);
    }

    fn render_block(&mut self, block: &Block) {
        //args: &RenderArgs

        use graphics::math::Matrix2d;
        use graphics::Transformed;

        let square_ = graphics::rectangle::Rectangle::new(block.colour).border(graphics::rectangle::Border {
            color: BLACK,
            radius: 0.01,
        });
        let dims_ =
            graphics::rectangle::rectangle_by_corners(0.0, 0.0, 2.0 / BOARD_WIDTH as f64, 2.0 / BOARD_HEIGHT as f64);
        let transform_: Matrix2d = graphics::math::identity()
            .trans(
                -((BOARD_WIDTH / 2) as f64) * 2.0 / BOARD_WIDTH as f64,
                (BOARD_HEIGHT / 2 - 1) as f64 * 2.0 / BOARD_HEIGHT as f64,
            )
            .trans(
                (block.position.x as f64) * 2.0 / BOARD_WIDTH as f64,
                -(block.position.y as f64) * 2.0 / BOARD_HEIGHT as f64,
            );
        let draw_state_ = graphics::draw_state::DrawState::default();
        square_.draw(dims_, &draw_state_, transform_, &mut self.gl);
    }
}