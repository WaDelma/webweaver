extern crate petgraph;
extern crate nalgebra;
extern crate num;
extern crate rand;
extern crate vec_map;

use std::ops::{Add, Sub, Mul, Div, IndexMut};
use std::marker::PhantomData;

use vec_map::VecMap;

use rand::{Rng, Rand, SeedableRng};

use nalgebra::{BaseFloat, Norm, Cast};

use num::{Zero};

use petgraph::Graph;
use petgraph::graph::{NodeIndex, IndexType};

const ELECTRIC_FORCE_CONSTANT: f64 = 8987551787.3681764;

pub struct Layout<P, F>(VecMap<P>, PhantomData<F>);

impl<P, F> Layout<P, F> {
    fn with_capacity(capacity: usize) -> Self {
        Layout(VecMap::with_capacity(capacity), PhantomData)
    }

    pub fn get(&self, index: NodeIndex) -> Option<&P> {
        self.0.get(index.index())
    }
}

impl<P, F> Layout<P, F>
    where P: Zero + IndexMut<usize, Output=F> + Norm<F> + Add<P, Output=P> + Sub<P, Output=P> + Mul<F, Output=P> + Div<F, Output=P> + Rand + Clone,
          F: BaseFloat,
{
    pub fn layout<N, E, SP>(g: &Graph<N, E>, spring_position: SP) -> Self
        where SP: Fn(NodeIndex, NodeIndex) -> P,
    {
        let mut coarce_stack = Vec::with_capacity(g.node_count());
        let mut cur = g.map(|_, _| (), |e, _| {
            let (source, target) = g.edge_endpoints(e).expect("When mapping edges their source and target nodes should exist.");
            (spring_position(source, target), spring_position(target, source))
        });
        while cur.edge_count() != 0 {
            let mut next = Graph::with_capacity(cur.node_count(), cur.edge_count());
            let mut mapping = VecMap::with_capacity(g.node_count());
            for node in 0..cur.node_count() {
                if mapping.contains_key(node) {
                    continue;
                }
                let new_node = next.add_node(());
                mapping.insert(node, new_node.index());
                let mut neighbors = cur.neighbors_undirected(NodeIndex::new(node));
                while let Some(neighbor) = neighbors.next() {
                    if neighbor.index() == node {
                        continue;
                    }
                    if !mapping.contains_key(neighbor.index()) {
                        mapping.insert(neighbor.index(), new_node.index());
                    }
                    for n in neighbors.chain(cur.neighbors_undirected(neighbor)) {
                        if neighbor == n || neighbor.index() == node || n.index() == node {
                            continue;
                        }
                        if let Some(&v) = mapping.get(n.index()) {
                            let edge = cur.find_edge_undirected(NodeIndex::new(node), n)
                                .unwrap_or_else(|| cur.find_edge_undirected(neighbor, n)
                                    .expect("If the edge wasn't one of nodes it should be it's neighbors edge.")).0;
                            let edge = cur.edge_weight(edge).expect("There should be edge weight for all existing edges.");
                            next.update_edge(new_node, NodeIndex::new(v), edge.clone());
                        }
                    }
                    break;
                }
            }
            // let name = format!("{}.dot", coarce_stack.len());
            // let mut file = std::fs::File::create(name).unwrap();
            // std::io::Write::write_all(&mut file, format!("{:?}", petgraph::dot::Dot::new(&cur)).as_bytes()).unwrap();
            // debug_assert_eq!(cur.node_count(), petgraph::visit::BfsIter::new(&petgraph::visit::AsUndirected(&cur), NodeIndex::new(0)).count());
            coarce_stack.push((cur, mapping));
            cur = next;
        }
        let mut layout = None;
        while let Some((cur, mapping)) = coarce_stack.pop() {
            layout = Some(Self::layout_level(&cur, coarce_stack.len() + 1, |i| {
                if let Some(lay) = layout.as_ref() {
                    let _: &Self = lay;
                    let i = mapping.get(i.index()).expect("Each levels mappings should contain valid mapping from nodes to their parents.");
                    lay.0.get(i.index()).expect("Latest layout should contain position of all nodes of that level.").clone()
                } else {
                    P::zero()
                }
            },
            |_, _| Cast::from(1.), //TODO: Spring length from amount of nodes combined?
            |from, to| {
                let edge = cur.find_edge_undirected(from, to).expect("Layouting one level should give us valid nodes that form an edge.").0;
                let edge = cur.edge_weight(edge).expect("There should be edge weight for all existing edges.");
                if from.index() < to.index() {
                    edge.0.clone()
                } else {
                    edge.1.clone()
                }
            }));
        }
        layout.expect("Coarcing should always contain atleast originals copy.")
    }

    pub fn layout_fast<N, E>(g: &Graph<N, E>) -> Self {
        Self::layout_level(g, 1, |_| P::zero(), |_, _| Cast::from(1.), |_, _| P::zero())
    }

    pub fn move_to_center(&mut self) {
        let mut middle = P::zero();
        for n in self.0.values() {
            middle = middle + n.clone() / Cast::from(self.0.len() as f64);
        }
        for (_, n) in self.0.iter_mut() {
            *n = n.clone() - middle.clone();
        }
    }

    fn layout_level<N, E, I, SL, SP>(g: &Graph<N, E>, level: usize, innitial: I, spring_length: SL, spring_position: SP) -> Layout<P, F>
        where I: Fn(NodeIndex) -> P,
              SL: Fn(NodeIndex, NodeIndex) -> F,
              SP: Fn(NodeIndex, NodeIndex) -> P,
    {
        let level = level as f64;
        let mut rand = rand::XorShiftRng::from_seed([1, 2, 3, 4]);
        let nodes = g.node_count();
        let mut layout = Layout::with_capacity(nodes);
        let mut temperature: F = Cast::from(level);
        let mut velocities = VecMap::with_capacity(nodes);
        let mut unstable = true;
        for node in 0..nodes {
            layout.0.insert(node, innitial(NodeIndex::new(node)));
            velocities.insert(node, P::zero());
        }
        while unstable {
            unstable = false;
            for n1 in 0..nodes {
                let vec = layout.0.get(n1)
                    .expect("Because nodes in petgraph are tighly packed there should be node for all indices in [0, nodes[").clone();
                let velocity = velocities.get(n1)
                    .expect("The velocity map should contain velocity value for all nodes.").clone();
                let mut acceleration = P::zero();
                for n2 in 0..nodes {
                    if n1 == n2 {
                        continue;
                    }
                    let cur_vec = layout.0.get(n2)
                        .expect("The layout should contain position for all nodes.").clone();
                    let diff = vec.clone() - cur_vec.clone();
                    let (unit, dist) = normalize_and_nudge(&mut rand, diff);

                    let q1 = 0.0001;
                    let q2 = 0.0001;
                    let magnitude: F = Cast::from(q1 * q2);
                    let ke: F = Cast::from(ELECTRIC_FORCE_CONSTANT);
                    let force = (unit / dist.powi(2)) * magnitude * ke;

                    acceleration = acceleration + force;
                }
                let n1 = NodeIndex::new(n1);
                for neighbor in g.neighbors_undirected(n1) {
                    let mut cur_vec = layout.0.get(neighbor.index())
                        .expect("The layout should contain position for all nodes including each nodes neighbor.").clone();
                    cur_vec = cur_vec + spring_position(n1, neighbor);
                    let vec = vec.clone() + spring_position(neighbor, n1);
                    let diff = cur_vec.clone() - vec.clone();
                    let (mut force, dist) = normalize_and_nudge(&mut rand, diff);
                    let spring = spring_length(n1, neighbor);
                    debug_assert!(spring == spring_length(neighbor, n1));

                    let springyness: F = Cast::from(24.);
                    force = force * (-springyness * (spring - dist));
                    let mut vel_diff = velocity.clone() - velocities.get(neighbor.index())
                        .expect("The velocity map should contain velocity value for all nodes including each nodes neighbor.").clone();
                    let half: F = Cast::from(0.5);
                    vel_diff = vel_diff * half;
                    let force = force - vel_diff;

                    acceleration = acceleration + force;
                }
                let velocity = velocity.clone() + acceleration;
                let (velocity, dist) = normalize_and_nudge(&mut rand, velocity);
                let velocity = velocity * temperature.min(dist);
                velocities.insert(n1.index(), velocity.clone());
                *layout.0.get_mut(n1.index()).expect("We already accessed this nodes position immutably.") = vec + velocity.clone();
                if velocity.norm() > Cast::from(level * 0.01) {
                    unstable = true;
                }
            }
            temperature = temperature * Cast::from(0.91);
        }
        layout
    }

    pub fn relayout(&mut self) {
        // TODO: If the graph is allowed to mutate, then we need to do magic here to prevent NodeIndices to be invalidated.
        // Only way I can think of is to recreate whole graph for Layout and add necessary information and use it as wrapper type.
        // This means that we should not depend on concrete graph type as we want to support custom graphs/wrappers too.
    }
}

fn normalize_and_nudge<R, P, F>(rand: &mut R, mut vec: P) -> (P, F)
    where P: Zero + IndexMut<usize, Output=F> + Norm<F> + Add<P, Output=P> + Sub<P, Output=P> + Mul<F, Output=P> + Div<F, Output=P> + Rand + Clone,
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
