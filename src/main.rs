#![allow(clippy::needless_return)] //ignore this rule, explicit returns are nicer for my brain.
mod float32;
mod float64;
mod forest;
mod node;
mod one_against_many;
mod reg_tree;
use float32::Float32;
#[allow(unused)]
use float64::Float64; // decide which Float wrapper to use


// this is to log how many trees are done training when training very large or many Forests
use lazy_static::lazy_static;
use std::sync::atomic::AtomicUsize;

// lazy_static! {
//     static ref GLOBAL_COUNTER: AtomicUsize = AtomicUsize::new(0);
// }
#[allow(unused)]
use ndarray_npy::read_npy;
#[allow(unused)] // needed for tests and debugging
use rand::seq::SliceRandom;

type R = Float32;
type G = Float32;

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;
    //G is type of feature data Vec<>
    // R is the predict return Type
    type R = Float32;
    type G = Float32;
}
