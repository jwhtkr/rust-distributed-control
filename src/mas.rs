//! This module contains the primary structs, types, and traits
//! for use in defining and analyzing a MAS.

use ndarray::{Array1, Array2};
use petgraph::graph::{NodeIndex, UnGraph};

/// Defines a Homogenous Multi-agent System with linear time-invariant dynamics
/// and an Undirected (but possibly weighted) communication graph.
/// I.e., every agent has dynamics $\dot{x}_i = A x_i + B u_i$.
/// This also calculates and stores the algebraic graph laplacian.
/// ```
/// use distributed_control::mas::HomMAS;
/// use petgraph::graph::UnGraph;
/// use ndarray::arr2;
///
/// let g = UnGraph::<f64, f64>::from_edges(&[(0, 1), (1, 2)]);
/// let a = arr2(&[[0., 0.], [0., 0.]]);
/// let b = arr2(&[[1.], [1.]]);
///
/// let sys = HomMAS::new(g, a, b).unwrap();
/// let n_agents = sys.n_agents;
/// println!("Number of agents is {n_agents}");
/// ```
#[derive(Debug)]
pub struct HomMAS {
    pub graph: UnGraph<f64, f64>,
    pub state_mat: Array2<f64>,
    pub input_mat: Array2<f64>,
    pub state_dim: usize,
    pub input_dim: usize,
    pub n_agents: usize,
    pub laplacian: Array2<f64>,
}

impl HomMAS {
    /// Create a new Homogenous Multi-agent System from an undirected
    /// communication graph, an LTI state matrix (A) and an LTI input
    /// matrix (B).
    pub fn new(
        graph: UnGraph<f64, f64>,
        state_mat: Array2<f64>,
        input_mat: Array2<f64>,
    ) -> Result<HomMAS, &'static str> {
        if state_mat.nrows() != state_mat.ncols() {
            return Err("The state matrix is not square.");
        }
        if state_mat.nrows() != input_mat.nrows() {
            return Err("The state dimension does not match the input row dimension.");
        }
        let state_dim = state_mat.ncols();
        let input_dim = input_mat.ncols();
        let n_agents = graph.node_count();
        let laplacian = degree(&graph) - adjacency(&graph);
        Ok(HomMAS {
            graph,
            state_mat,
            input_mat,
            state_dim,
            input_dim,
            n_agents,
            laplacian,
        })
    }
}

/// Generate the weighted degree matrix of a communication graph.
pub fn degree(graph: &UnGraph<f64, f64>) -> Array2<f64> {
    Array2::from_diag(&adjacency(graph).dot(&Array1::ones((graph.node_count(),))))
}

/// Generate the weighted adjacency matrix of the communication graph.
pub fn adjacency(graph: &UnGraph<f64, f64>) -> Array2<f64> {
    Array2::from_shape_fn((graph.node_count(), graph.node_count()), |(i, j)| {
        get_edge_weight(&graph, i, j)
    })
}

/// Generate the incidence matrix of the communication graph.
pub fn incidence(graph: &UnGraph<f64, f64>) -> Array2<f64> {
    let mut out = Array2::zeros((graph.node_count(), graph.edge_count()));
    for edge_idx in graph.edge_indices() {
        let (a, b) = graph.edge_endpoints(edge_idx).unwrap();
        out[(a.index(), edge_idx.index())] = -1.;
        out[(b.index(), edge_idx.index())] = 1.;
    }
    return out;
}

/// Generate the weighted edge matrix of the communication graph.
pub fn edge_weight(graph: &UnGraph<f64, f64>) -> Array2<f64> {
    Array2::from_diag(&Array1::from_iter(graph.edge_indices().map(|i| *graph.edge_weight(i).unwrap())))
}

fn get_edge_weight(graph: &UnGraph<f64, f64>, a: usize, b: usize) -> f64 {
    match graph
        .edges_connecting(NodeIndex::new(a), NodeIndex::new(b))
        .next()
    {
        Some(edge) => *edge.weight(),
        None => 0.,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray;
    use petgraph;

    fn make_test_g() -> petgraph::graph::UnGraph<f64, f64> {
        let mut g = petgraph::graph::UnGraph::<f64, f64>::with_capacity(3, 2);
        g.add_node(1.);
        g.add_node(1.);
        g.add_node(1.);
        g.add_edge(
            petgraph::graph::node_index(0),
            petgraph::graph::node_index(1),
            1.,
        );
        g.add_edge(
            petgraph::graph::node_index(1),
            petgraph::graph::node_index(2),
            1.,
        );
        return g;
    }

    #[test]
    fn test_mas_creation() {
        let a = ndarray::arr2(&[[0.]]);
        let b = ndarray::arr2(&[[1.]]);
        let sys = HomMAS::new(make_test_g(), a, b).unwrap();
        assert_eq!(sys.state_dim, 1);
        assert_eq!(sys.input_dim, 1);
        assert_eq!(sys.n_agents, 3);
        assert_eq!(
            degree(&sys.graph),
            ndarray::arr2(&[[1., 0., 0.], [0., 2., 0.,], [0., 0., 1.]])
        );
        assert_eq!(
            adjacency(&sys.graph),
            ndarray::arr2(&[[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
        );
        assert_eq!(
            sys.laplacian,
            ndarray::arr2(&[[1., -1., 0.], [-1., 2., -1.], [0., -1., 1.]])
        );
        assert_eq!(
            incidence(&sys.graph),
            ndarray::arr2(&[[-1., 0.], [1., -1.], [0., 1.]])
        );
        assert_eq!(edge_weight(&sys.graph), ndarray::arr2(&[[1., 0.], [0., 1.]]));
    }

    #[test]
    #[should_panic(expected = "does not match")]
    fn test_mas_creation_state_input_mismatch() {
        let a = ndarray::arr2(&[[0.]]);
        let b = ndarray::arr2(&[[1.], [1.]]);
        let sys = HomMAS::new(make_test_g(), a, b);
        sys.unwrap();
    }

    #[test]
    #[should_panic(expected = "is not square")]
    fn test_mas_creation_state_not_square() {
        let a = ndarray::arr2(&[[0.], [0.]]);
        let b = ndarray::arr2(&[[1.], [1.]]);
        let sys = HomMAS::new(make_test_g(), a, b);
        sys.unwrap();
    }
}
