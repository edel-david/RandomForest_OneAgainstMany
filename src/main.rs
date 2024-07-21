#![allow(unused)]
mod utils;
mod node;
mod float64;
mod reg_tree;
mod forest;
use reg_tree::RegressionTree;
use node::Node;
use float64::Float64;
//extern crate supa_cube_forest;
//use crate::utils::*;
use core::hash::Hash;
use core::ops::Add;
use std::cmp::Ordering;
use std::fmt::{Debug, Error};
use std::hash::Hasher;
use std::iter::zip;
use std::ops::{Deref, Div, Mul, Neg, Rem, Sub};
use std::str::FromStr;
use std::{mem, usize};
//use core::slice::SlicePattern;
use conv::prelude::*;
use ndarray::{prelude::*, ScalarOperand, ViewRepr};
use ndarray_npy::{read_npy, ReadableElement};
use ndarray_stats::SummaryStatisticsExt;
use num::{Float, FromPrimitive, NumCast, One, Signed, ToPrimitive, Zero};
use rand::seq::SliceRandom;
use std::collections::HashSet;
use std::{collections::HashMap, hash};
use std::{f64::INFINITY, marker::PhantomData};



fn main() {
    type G = Float64;
    type T = i64;
    type R = Float64;

    let size = mem::size_of::<Node<R, G>>();
    println!("Size of MyEnum: {} bytes", size);

    // let data: Array2<f64> = read_npy("src/digits_data.npy").expect("file is present and correct");
    // let data = data.mapv(|x| Float64(x));
    // let target: Array1<i64> =
    //     read_npy("src/digits_target.npy").expect("file is present and correct");

    // three9 is 363 long
    let data: Array2<f64> = read_npy("src/three_9_data.npy").expect("file is present and correct");
    //let data = array![[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]];
    let data = data.mapv(|x| Float64(x as f64));
    let target: Array1<i64> =
        read_npy("src/three_9_target.npy").expect("file is present and correct");
    dbg!(target.sum());

    let mut tree: RegressionTree<G, R> = RegressionTree {
        root: None,
        n_min: 10,
        d_try: 8,
    };

    let mut all_indices:Vec<_> = (0..363).collect();
    let mut rng = ndarray_rand::rand::thread_rng();
    all_indices.shuffle(&mut rng);
    let train_indices = all_indices[73..].to_vec();
    let test_indices = all_indices[..73].to_vec(); 
    let ts = data.select(Axis(0),&train_indices);
    let ts_target = target.select(Axis(0),&train_indices);
    tree.train(&ts, &ts_target, Some(8));
    
    let test_set_feat = data.select(Axis(0),&test_indices);
    let test_set_resp = target.select(Axis(0),&test_indices);


    let perf = tree.evaluate(&test_set_feat, &test_set_resp);
    dbg!(perf);

}
// T is the type of the response values
//G is type of feature data Vec<>
// R is the predict return Type
