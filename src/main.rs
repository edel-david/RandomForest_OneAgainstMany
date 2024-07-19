use std::marker::PhantomData;

use ndarray::prelude::*;
use ndarray::Array;
use ndarray_rand;
use ndarray_rand::RandomExt;
use rand::seq::SliceRandom;
use ndarray_npy::read_npy;
fn main() {
    // load data and target
    //const dim :Dim<[usize; 2]> = Dim([1797,64]);
    let data:Array2<f64>=read_npy("src/digits_data.npy").expect("file is present and correct");
    let target:Array1<i64> = read_npy("src/digits_target.npy").expect("file is present and correct");

    println!("{:?}",data.dim());
    println!("{:?}",target.dim());


    return ()
}
// T is the type of the prediction value
//G is type of feature data Vec<>
enum Node<T: Clone, G: Clone> {
    SplitNode {
        left: Box<Node<T, G>>,
        right: Box<Node<T, G>>,
        split_index: usize,
        threashold: G,
    },
    LeafNode {
        prediction: T,
    },
}

//G is type of feature data Vec<>
struct RegressionTree<T: Clone, G: Clone> {
    root: Box<Node<T, G>>,
    n_min: usize,
    input_data: PhantomData<G>,
}

impl<T: Clone, G: Clone + std::cmp::PartialOrd> RegressionTree<T, G>
where f64: From<G>{
    fn predict(&self, x: Vec<G>) -> T {
        let mut node = &(self.root);
        for _i in 0..100 {
            match node.as_ref() {
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
            }
        }
        todo!("predict followed split node for over 100 splits! not possible?");
    }
    /// j is feature index and t is threashold
    fn compute_loss_for_split(self,features: &Vec<&Vec<G>>,responses: Vec<T>,j:usize,t:f64) {
        let arr = Array1::from(features[0].clone());
        let var = arr.mapv(|x| f64::from(x)).var(0.0);
        todo!();
    }

    /// return a 1-D array with D_try randomly selected indices from 0...(D-1).
    fn select_active_indices(D: usize, D_try: usize) -> Vec<usize> {
        let mut rng = ndarray_rand::rand::thread_rng();
        let mut myvec: Vec<usize> = (0..D).collect();
        myvec.shuffle(&mut rng);
        return myvec[..D_try].to_vec();
    }

    fn find_threashold(self,features: &Vec<&Vec<G>>, responses: &Vec<T>,j:usize){
        // return: a 1-D array with all possible thresholds along feature j
        //(find midpoints between instances of feature j)
        let array = Array::from_vec(features.clone());
        let temp = array.select(Axis(1),&[j]);

    }
    fn train(self, features: &Vec<&Vec<G>>, responses: &Vec<T>, mut D_try: Option<usize>) {
        let N = features.len();
        let D = features
            .first()
            .expect("An empty train set does not make sense")
            .len();
        let D_try = match D_try {
            None => 8,
            Some(val) => val,
        };

        while true {
            let active_indices = RegressionTree::<T, G>::select_active_indices(D, D_try);
            
        }
        todo!()
    }
}

