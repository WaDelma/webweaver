extern crate petgraph;
extern crate nalgebra;
extern crate num;
extern crate rand;

use std::ops::{Add, Sub, Mul, Div};
use std::collections::HashMap;
use std::marker::PhantomData;

use rand::{Rng, Rand, SeedableRng};

use nalgebra::{BaseFloat, Norm, Cast};

use num::Zero;

use petgraph::Graph;
use petgraph::graph::{NodeIndex, IndexType};

const ELECTRIC_FORCE_CONSTANT: f64 = 8987551787.3681764;

pub struct GraphLayout<P, F>(HashMap<usize, P>, PhantomData<F>);

impl<P, F> GraphLayout<P, F> {
    pub fn get(&self, index: NodeIndex) -> Option<&P> {
        self.0.get(&index.index())
    }
}

impl<P, F> GraphLayout<P, F>
    where P: Zero + Norm<F> + Add<P, Output=P> + Sub<P, Output=P> + Mul<F, Output=P> + Div<F, Output=P> + Rand + Clone,
          F: BaseFloat,
{
    pub fn layout<N, E, SL, SP>(g: &Graph<N, E>, spring_length: SL, spring_position: SP) -> Self
        where SL: Fn(NodeIndex, NodeIndex) -> F,
              SP: Fn(NodeIndex, NodeIndex) -> P,
    {
        let mut rand = rand::thread_rng();
        let mut layout = Self::new();
        let nodes = g.node_count();
        let level = nodes;
        let mut temperature: F = Cast::from(level as f64);
        let mut velocities = HashMap::new();
        let mut stabilised = false;
        for node in 0..nodes {
            layout.0.insert(node, P::zero());
            velocities.insert(node, P::zero());
        }
        while !stabilised {
            let mut flag = true;
            for n1 in 0..nodes {
                let vec = layout.0.get(&n1).unwrap().clone();
                let velocity = velocities.get(&n1).unwrap();
                let mut acceleration = P::zero();
                for n2 in 0..nodes {
                    if n1 == n2 {
                        continue;
                    }
                    let cur_vec = layout.0.get(&n2).unwrap().clone();
                    let diff = vec.clone() - cur_vec.clone();
                    let (unit, dist) = normalize_and_nudge(&mut rand, diff);

                    let q1 = 0.0001;//0.0000000000000000001602;
                    let q2 = 0.0001;//0.0000000000000000001602;
                    let force = (unit / dist.powi(2)) * Cast::from(q1 * q2) * Cast::from(ELECTRIC_FORCE_CONSTANT);

                    // let mul: F = Cast::from(-0.2 * level as f64 * level as f64);
                    // let force = unit * mul / dist;

                    acceleration = acceleration + force;
                }
                let n1 = NodeIndex::new(n1);
                for neighbor in g.neighbors_undirected(n1) {
                    let mut cur_vec = layout.0.get(&neighbor.index()).unwrap().clone();
                    cur_vec = cur_vec + spring_position(n1, neighbor);
                    let vec = vec.clone() + spring_position(neighbor, n1);
                    let diff = cur_vec.clone() - vec.clone();
                    let (mut force, dist) = normalize_and_nudge(&mut rand, diff);
                    let spring = spring_length(n1, neighbor);
                    debug_assert!(spring == spring_length(neighbor, n1));

                    let springyness: F = Cast::from(24.);
                    let spring_length: F = Cast::from(1.);
                    force = force * (-springyness * (spring_length - dist));
                    let mut vel_diff = velocity.clone() - velocities.get(&neighbor.index()).unwrap().clone();
                    vel_diff = vel_diff * Cast::from(0.5);
                    let force = force - vel_diff;

                    // force = force * dist.powi(2) / spring;
                    acceleration = acceleration + force;
                }
                let dist = acceleration.norm();
                acceleration = acceleration.normalize();
                acceleration = acceleration * temperature.min(dist);
                *layout.0.get_mut(&n1.index()).unwrap() = vec + acceleration.clone();
                if acceleration.norm() > Cast::from(level as f64 * 0.01) {
                    flag = false;
                }
            }
            stabilised = flag;
            temperature = temperature * Cast::from(0.91);
        }
        let mut middle = P::zero();
        for n in layout.0.values() {
            middle = middle + n.clone() / Cast::from(nodes as f64);
        }
        for (_, n) in layout.0.iter_mut() {
            *n = n.clone() - middle.clone();
        }
        layout
    }

    pub fn relayout(&mut self) {
        // TODO: If the graph is allowed to mutate, then we need to do magic here to prevent NodeIndices to be invalidated.
        // Only way I can think of is to recreate whole graph for GraphLayout and add necessary information and use it as wrapper type.
        // This means that we should not depend on concrete graph type as we want to support custom graphs/wrappers too.
    }

    fn new() -> Self {
        GraphLayout(HashMap::new(), PhantomData)
    }
}

fn normalize_and_nudge<R, P, F>(rand: &mut R, mut vec: P) -> (P, F)
    where P: Zero + Norm<F> + Add<P, Output=P> + Sub<P, Output=P> + Mul<F, Output=P> + Div<F, Output=P> + Rand + Clone,
          F: BaseFloat,
          R: Rng,
{
    let mut dist = vec.norm();
    let min = Cast::from(0.0001);
    if dist < min {
        vec = P::rand(rand) * min;
        dist = min;
    } else {
        vec = vec / dist;
    }
    (vec, dist)
}
