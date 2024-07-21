#[derive(Debug)]
pub enum Node<R: Clone + num::Zero, G: Clone> {
    SplitNode {
        left: Option<Box<Node<R, G>>>,
        right: Option<Box<Node<R, G>>>,
        split_index: usize,
        threashold: G,
    },
    LeafNode {
        prediction: R,
    },
}