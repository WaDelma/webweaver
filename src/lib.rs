extern crate petgraph;
extern crate nalgebra;
extern crate num;
extern crate rand;

use std::ops::{Add, Sub, Mul, Div};
use std::collections::HashMap;
use std::marker::PhantomData;

use rand::Rand;

use nalgebra::{BaseFloat, Norm, Cast};

use num::Zero;

use petgraph::Graph;
use petgraph::graph::{NodeIndex, IndexType};

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
    pub fn layout<N, E, SL, SP>(g: &Graph<N, E>, mut spring_length: SL, mut spring_position: SP) -> Self
        where SL: FnMut(NodeIndex, NodeIndex) -> F,
              SP: FnMut(NodeIndex, NodeIndex) -> P,
    {
        let mut rand = rand::thread_rng();
        let mut layout = Self::new();
        let nodes = g.node_count();
        let level = nodes;
        let mut temperature: F = Cast::from(level as f64);
        let mut stabilised = false;
        for node in 0..nodes {
            layout.0.insert(node, P::rand(&mut rand));
        }
        while !stabilised {
            let mut flag = true;
            for n1 in 0..nodes {
                let vector = layout.0.get(&n1).unwrap().clone();
                let mut displacement = P::zero();
                for n2 in 0..nodes {
                    if n1 == n2 {
                        continue;
                    }
                    let cur_vec = layout.0.get(&n2).unwrap().clone();
                    let mut local_vec = cur_vec.clone() - vector.clone();
                    let mut dist = local_vec.norm();
                    if dist == F::zero() {
                        let cur_vec = cur_vec + P::rand(&mut rand);
                        layout.0.insert(n2, cur_vec.clone());
                        local_vec = cur_vec - vector.clone();
                        dist = local_vec.norm();
                    }
                    local_vec = local_vec.normalize();
                    let mul: F = Cast::from(-0.2 * level as f64 * level as f64);
                    local_vec = local_vec * mul / dist;
                    displacement = displacement + local_vec;
                }
                let n1 = NodeIndex::new(n1);
                for neighbor in g.neighbors_undirected(n1) {
                    let mut cur_vec = layout.0.get(&neighbor.index()).unwrap().clone();
                    cur_vec = cur_vec + spring_position(n1, neighbor);
                    let vector = vector.clone() + spring_position(neighbor, n1);
                    let mut local_vec = cur_vec.clone() - vector.clone();
                    let mut dist = local_vec.norm();
                    if dist == F::zero() {
                        let cur_vec = cur_vec + P::rand(&mut rand);
                        layout.0.insert(neighbor.index(), cur_vec.clone());
                        local_vec = cur_vec - vector;
                        dist = local_vec.norm();
                    }
                    local_vec = local_vec / dist;
                    local_vec = local_vec * dist * dist / spring_length(n1, neighbor);
                    displacement = displacement + local_vec;
                }
                let dist = displacement.norm();
                displacement = displacement.normalize();
                displacement = displacement * temperature.min(dist);
                *layout.0.get_mut(&n1.index()).unwrap() = vector + displacement.clone();
                if displacement.norm() > Cast::from(level as f64 * 0.01) {
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
