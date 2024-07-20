#![allow(unused)]
mod utils;
//extern crate supa_cube_forest;
//use crate::utils::*;
use core::hash::Hash;
use core::ops::Add;
use std::cmp::Ordering;
use std::fmt::{Debug, Error};
use std::hash::Hasher;
use std::ops::{Deref, Div, Mul, Neg, Rem, Sub};
use std::str::FromStr;
use std::{mem, usize};
//use core::slice::SlicePattern;
use conv::prelude::*;
use ndarray::{prelude::*, ScalarOperand, ViewRepr};
use ndarray_npy::{read_npy, ReadableElement};
use num::{Float, FromPrimitive, NumCast, One, ToPrimitive, Zero};
use rand::seq::SliceRandom;
use std::collections::HashSet;
use std::{collections::HashMap, hash};
use std::{f64::INFINITY, marker::PhantomData};
#[derive(Debug, PartialEq, PartialOrd, Copy, Clone)]
pub struct Float64(f64);
#[derive(Debug)]
pub enum Node<'a, R: Clone + num::Zero, G: Clone> {
    SplitNode {
        left: Option<Box<Node<'a, R, G>>>,
        right: Option<Box<Node<'a, R, G>>>,
        split_index: usize,
        threashold: G,
    },
    LeafNode {
        prediction: R,
    },
    UnknownNode {
        features:&'a Array1<G>,
        targets:&'a Array1<G>
    }
}



fn main() {
    type G = Float64;
    type T = i64;
    type R = Float64;
    
    let size = mem::size_of::<Node<R,G>>();
    println!("Size of MyEnum: {} bytes", size);

    
    // let data: Array2<f64> = read_npy("src/digits_data.npy").expect("file is present and correct");
    // let data = data.mapv(|x| Float64(x));
    // let target: Array1<i64> =
    //     read_npy("src/digits_target.npy").expect("file is present and correct");
    
    let data: Array2<f64> = read_npy("src/three_9_data.npy").expect("file is present and correct");
    let data = array![[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]];
    let data = data.mapv(|x| Float64(x as f64));
    let target: Array1<i64> =
    read_npy("src/three_9_target.npy").expect("file is present and correct");
    dbg!(target.sum());
    let target = array![-1,-1,1,1];

    let mut tree: RegressionTree<G, R> = RegressionTree {
        root: None,
        n_min: 2,
        d_try:2
    };


    tree.train(&data,&target,None);
    dbg!(&tree);
    println!("{:?}", data.dim());
    println!("{:?}", target.dim());

    let mut results = Vec::new();
    for row in data.rows() {
        results.push(tree.predict(&row.to_owned()))
    }
    dbg!(&results);

}
// T is the type of the response values
//G is type of feature data Vec<>
// R is the predict return Type
#[derive(Debug)]
pub struct RegressionTree<'a,G: Clone, R>
where
    G: From<f64> + FromPrimitive + Float, //From<usize>,
    R: Clone + num::Zero,
{
    root: Option<Box<Node<'a,R, G>>>,
    n_min: usize,
    d_try:usize
    //input_data: PhantomData<G>,
}

impl<'a,G: Clone + std::cmp::PartialOrd, R> RegressionTree<'a,G, R>
where
    G: Add
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
    R: Clone + num::Zero + From<f64>+From<usize> + std::ops::Div<Output = R>,
    //T: std::convert::From<f64> + num::Zero,
{
    pub fn predict(&self, x: &Array1<G>) -> R {
        let mut node = &(self.root);
        for _i in 0..100 {
            match node.as_ref() {
                Some(val) => match val.as_ref() {
                    Node::LeafNode { prediction } => {
                        return prediction.clone();
                    }
                    Node::SplitNode {
                        left,
                        right,
                        split_index,
                        threashold,
                    } => {
                        if x[*split_index] <= *threashold {
                            node = left
                        } else {
                            node = right
                        }
                    }
                    Node::UnknownNode { features, targets }
                => {return 0_f64.into()}
                },
                None => {
                    return R::from(0_f64); // if not trained yet, return 0 always
                                           // if trained but still reached => bug
                                           // consider logging here, as it does not make sense to use predict before training
                }
            }
        }

        todo!("predict followed split node for over 100 splits! not possible?");
    }
    /// j is feature index and t is threashold
    fn compute_loss_for_split<T: Clone + Into<G>>(
        &self,
        features: &Array2<G>,
        responses: &Array1<T>,
        j: usize,
        t: G,
    ) -> G {
        //let arr = Array1::from(features[0].clone());
        let temp = features.slice(s![.., j]);
        let left_indices = temp.mapv(|x: G| (x <= t) as usize);
        let right_indices = left_indices.mapv(|x| 1 - x);
        let amount_left = (left_indices.sum());
        let amount_right = (right_indices.sum());
        if (amount_left < self.n_min) | (amount_right < self.n_min) {
            return <G as From<f64>>::from(INFINITY);
        }
        let left = responses
            .select(Axis(0), left_indices.as_slice().expect("idk if this works"))
            .mapv(|x| x.into());

        let right = responses
            .select(
                Axis(0),
                right_indices.as_slice().expect("idk if this works"),
            )
            .mapv(|x| x.into());
        let zero = <G as From<f64>>::from(0_f64);
        let loss = left.var(zero) * G::value_from(amount_left).expect("msg")// into G
            + right.var(zero) * G::value_from(amount_right).expect("msg");
        return loss;
    }

    /// return a 1-D array with D_try randomly selected indices from 0...(D-1).
    fn select_active_indices(D: usize, d_try: usize) -> Vec<usize> {
        let mut rng = ndarray_rand::rand::thread_rng();
        let mut myvec: Vec<usize> = (0..D).collect();
        myvec.shuffle(&mut rng);
        myvec[..d_try].to_vec()
    }
    fn find_threasholds<T>(
        &self,
        features: &Array2<G>,
        responses: &Array1<T>,
        j: usize,
    ) -> Array1<G> {
        // return: a 1-D array with all possible thresholds along feature j
        //(find midpoints between instances of feature j)
        let mut feature_slice = features.select(Axis(1), &[j]);
        let unique_feature: HashSet<_> = feature_slice.iter().cloned().collect();
        let mut unique_feature_array = Array1::from_iter(unique_feature);
        unique_feature_array
            .as_slice_mut()
            .expect("idk if this workds")
            .sort();
        let temp2 = unique_feature_array.clone().select(
            Axis(0),
            Vec::from_iter(1..(unique_feature_array.dim())).as_slice(),
        );
        let mut new = unique_feature_array.slice_mut(s![..unique_feature_array.len()-1]) .add(temp2);
        //dbg!(&new);
        let new = new / <G as From<u8>>::from(2);
        //dbg!(&new);
        return new;
    }
    pub fn create_correct_node<T: Clone + Into<G> + Into<R> + num::Zero>(&self,features: &Array2<G>,responses: &Array1<T>)->Node<'a,R, G>{
        #[allow(non_snake_case)] // to allow mathematical notation
        let N = features.dim().0;
        #[allow(non_snake_case)]
        let D = features.dim().1;
        let mut l_min = <G as From<f64>>::from(INFINITY);
            let mut j_min = 0;
            let mut t_min = <G as From<f64>>::from(0_f64);
            let active_indices = RegressionTree::<G, R>::select_active_indices(D, self.d_try);
            for j in active_indices {
                let threasholds =
                    RegressionTree::<G, R>::find_threasholds(&self, &features, responses, j);
                for threashold in threasholds {
                    let loss = RegressionTree::<G, R>::compute_loss_for_split(
                        self, features, responses, j, threashold,
                    );
                    if loss < l_min {
                        l_min = loss;
                        j_min = j;
                        t_min = threashold;
                    }
                }
            }
            if l_min == <G as From<f64>>::from(INFINITY) {
                // we did not find a threshold, so this should become a leaf node
                let sum = responses.sum() ;
                let mut sum_r: R = sum.into() ;
                let len_r :R= responses.dim().into();
                sum_r = sum_r / len_r;
                return Node::LeafNode::<R, G> { prediction: sum_r };

            }
            else {
                // threashold exists => split node

                // get left and right
                let temp_to_split = features.slice(s![.., j_min]);
                let left_indices = temp_to_split.mapv(|x| (x <= t_min) as usize);
                let right_indices = left_indices.mapv(|x| 1 - x);
                let left_feat = features.select(Axis(0), left_indices.as_slice().expect("msg"));
                let right_feat = features.select(Axis(0), right_indices.as_slice().expect("msg"));
                let left_resp = responses.select(Axis(0), left_indices.as_slice().expect("msg"));
                let right_resp = responses.select(Axis(0), right_indices.as_slice().expect("msg"));

                let left = Self::create_correct_node(&self,&left_feat,&left_resp);
                let right = Self::create_correct_node(&self,&right_feat,&right_resp);
                return Node::SplitNode::<'a,R, G> {
                    left: Some(Box::new(left)),
                    right: Some(Box::new(right)),
                    split_index: j_min,
                    threashold: t_min,
                };
            }
        
        todo!()}
    pub fn train<T: Clone + Into<G> + Into<R> + num::Zero>(
        &mut self,
        features: &Array2<G>,
        responses: &Array1<T>,
        d_try: Option<usize>,
    ) -> Result<String, String> {
        #[allow(non_snake_case)] // to allow mathematical notation
        let N = features.dim().0;
        #[allow(non_snake_case)]
        let D = features.dim().1;
        let d_try = d_try.unwrap_or(self.d_try);

        //let working_node = &mut self.root;

        let mut l_min = <G as From<f64>>::from(INFINITY);
        let mut j_min = 0;
        let mut t_min = <G as From<f64>>::from(0_f64);

        l_min = <G as From<f64>>::from(INFINITY);
        j_min = 0;
        t_min = <G as From<f64>>::from(0_f64);
        let active_indices = RegressionTree::<G, R>::select_active_indices(D, d_try);
        for j in active_indices {
            let threasholds =
                RegressionTree::<G, R>::find_threasholds(&self, &features, responses, j);
            for threashold in threasholds {
                let loss = RegressionTree::<G, R>::compute_loss_for_split(
                    self, features, responses, j, threashold,
                );
                if loss < l_min {
                    l_min = loss;
                    j_min = j;
                    t_min = threashold;
                }
            }
        }
        let node = if (l_min == <G as From<f64>>::from(INFINITY)) {
            // we did not find a threshold, so this should become a leaf node
            // append leaf node
            //root
            let sum = responses.sum() ;
            let mut sum_r: R = sum.into() ;
            let len_r :R= responses.dim().into();
            sum_r = sum_r / len_r;
            Node::LeafNode::<R, G> { prediction: sum_r }
        } else {
            // ELSE : assume a split has been found.
            // Make first Node:


            // make left and right
            let temp_to_split = features.slice(s![.., j_min]);
            let left_indices = temp_to_split.mapv(|x| (x <= t_min) as usize);
            let right_indices = left_indices.mapv(|x| 1 - x);
            let left_feat = features.select(Axis(0), left_indices.as_slice().expect("msg"));
            let right_feat = features.select(Axis(0), right_indices.as_slice().expect("msg"));
            let left_resp = responses.select(Axis(0), left_indices.as_slice().expect("msg"));
            let right_resp = responses.select(Axis(0), right_indices.as_slice().expect("msg"));
            let left = Self::create_correct_node(&self,&left_feat,&left_resp);
            let right = Self::create_correct_node(&self,&right_feat,&right_resp);

            Node::SplitNode::<R, G> {
                left: Some(Box::new(left)),
                right: Some(Box::new(right)),
                split_index: j_min,
                threashold: t_min,
            }
        };
        self.root = Some(Box::new(node));
        return Ok(String::from("sucess"));
        todo!()
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test(){
        type G = Float64;
        type T = i64;
        type R = Float64;
        
        let size = mem::size_of::<Node<R,G>>();
        println!("Size of MyEnum: {} bytes", size);
    
        
        // let data: Array2<f64> = read_npy("src/digits_data.npy").expect("file is present and correct");
        // let data = data.mapv(|x| Float64(x));
        // let target: Array1<i64> =
        //     read_npy("src/digits_target.npy").expect("file is present and correct");
        
        let data: Array2<f64> = read_npy("src/three_9_data.npy").expect("file is present and correct");
        let data = array![[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]];
        let data = data.mapv(|x| Float64(x as f64));
        let target: Array1<i64> =
        read_npy("src/three_9_target.npy").expect("file is present and correct");
        let target = array![-1,-1,1,1];
    
        let mut tree: RegressionTree<G, R> = RegressionTree {
            root: None,
            n_min: 2,
            d_try:2
        };
    
    
        tree.train(&data,&target,None);
        dbg!(&tree);
        println!("{:?}", data.dim());
        println!("{:?}", target.dim());
    
        let mut results = Vec::new();
        for row in data.rows() {
            results.push(tree.predict(&row.to_owned()))
        }
        dbg!(&results);
    }
}