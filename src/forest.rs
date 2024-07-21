use conv::ValueFrom;
use ndarray::Array1;
use num::{Float, FromPrimitive, Signed};
use rand::{thread_rng, Rng};
use crate::float64::Float64;
use crate::reg_tree::RegressionTree;
use ndarray::prelude::*;
use std::fmt::Debug;
use std::ops::Add;
use core::hash::Hash;

// type G = Float64; // G is type of feature data Vec<>
// type T = i64; // T is the type of the response values
// type R = Float64; // R is the predict return Type

struct Forest<G,R> 
where 
G:From<f64> + FromPrimitive + Float,
R:Clone + num::Zero
{
    amount_trees:usize,
    trees:Vec<RegressionTree<G,R>>,
    n_min:usize,
    dtry:usize,
}


impl<G,R> Forest<G,R> 
where 
G: Add
+ From<usize>
+ Debug
+ ValueFrom<usize>
+ From<f64>
+ Hash
+ FromPrimitive
+ Float
+ From<u8>
+ From<R>
+ Ord
+ ndarray::ScalarOperand
+ std::ops::Add<Output = G>
+ core::ops::Add
+ Copy,
R: Clone
+ num::Zero
+ From<f64>
+ From<usize>
+ std::ops::Div<Output = R>
+ std::ops::Sub<Output = R>
+ std::cmp::PartialOrd
+ Signed,
{
    fn new(amount_trees:usize,n_min:usize,D_try:Option<usize>){}

    fn train<T:Clone + Into<R> + Into<G> + num::Zero + Debug>(&mut self,features:&Array2<G>,responses:&Array1<T>){
    let n = responses.dim();
    for tree in &mut self.trees {
        let indices = bootstrap_sampling(n);
        let to_train = features.select(Axis(0),&indices);
        let to_train_resp = responses.select(Axis(0), &indices);
        tree.train(&to_train,&to_train_resp,None);
    }
}

fn predict(&self,features:&Array1<G>) {
    let mut results: Vec<R> = Vec::with_capacity(self.amount_trees);
    for tree in &self.trees {
        results.push(tree.predict(features))
    }
}

}

fn bootstrap_sampling(n:usize)->Vec<usize> {
    let mut rng = thread_rng();
    let indices:Vec<usize> = (0..n).map(|_|rng.gen_range(0..n)).collect();
    return indices;
}