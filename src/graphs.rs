//! This module contains abstractions for algebraic graph concepts.
//! E.g., adjacency, degree, laplacian, incidence, and edge weight
//! matrices.

use ndarray::{Array1, Array2, LinalgScalar};
use petgraph::{graph::NodeIndex, visit::EdgeRef, EdgeType, Graph};

/// Generate the weighted adjacency matrix of the communication graph.
pub fn adjacency<N, E, Ty>(graph: &Graph<N, E, Ty>) -> Array2<E>
where
    E: LinalgScalar,
    Ty: EdgeType,
{
    let adj =Array2::from_shape_fn((graph.node_count(), graph.node_count()), |(i, j)| {
        *get_edge_weight(&graph, i, j).unwrap_or(&E::zero())
    });
    if graph.is_directed() {
        return adj.t().to_owned();
    }
    adj
}

/// Generate the weighted degree matrix of a communication graph.
pub fn degree<N, E, Ty>(graph: &Graph<N, E, Ty>) -> Array2<E>
where
    E: LinalgScalar,
    Ty: EdgeType,
{
    Array2::from_diag(&adjacency(graph).dot(&Array1::ones((graph.node_count(),))))
}

///Generate the Laplacian matrix of the communication graph.
pub fn laplacian<N, E, Ty>(graph: &Graph<N, E, Ty>) -> Array2<E>
where
    E: LinalgScalar,
    Ty: EdgeType,
{
    degree(graph) - adjacency(graph)
}

/// Generate the incidence matrix of the communication graph.
pub fn incidence<N, E, Ty>(graph: &Graph<N, E, Ty>) -> Array2<E>
where
    E: LinalgScalar,
    Ty: EdgeType,
{
    let mut out: Array2<E> = Array2::zeros((graph.node_count(), graph.edge_count()));
    for edge_idx in graph.edge_indices() {
        let (a, b) = graph.edge_endpoints(edge_idx).unwrap();
        out[(a.index(), edge_idx.index())] = E::zero() - E::one();
        out[(b.index(), edge_idx.index())] = E::one();
    }
    out
}

/// Generate the weighted edge matrix of the communication graph.
pub fn edge_weight<N, E, Ty>(graph: &Graph<N, E, Ty>) -> Array2<E>
where
    E: LinalgScalar,
    Ty: EdgeType,
{
    Array2::from_diag(&Array1::from_iter(
        graph.edge_indices().map(|i| *graph.edge_weight(i).unwrap()),
    ))
}

/// Get the weight for the edge connecting the node at index `a` to the node at index `b`.
fn get_edge_weight<N, E: LinalgScalar, Ty: EdgeType>(
    graph: &Graph<N, E, Ty>,
    a: usize,
    b: usize,
) -> Option<&E> {
    graph
        .edges(NodeIndex::new(a))
        .filter(|e| e.target() == NodeIndex::new(b))
        .map(|e| e.weight())
        .next()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray;
    use petgraph;

    fn undirected_unweighted_line_graph() -> petgraph::graph::UnGraph<f64, f64> {
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
    fn test_degree() {
        let d_mat = degree(&undirected_unweighted_line_graph());
        assert_eq!(
            d_mat,
            ndarray::Array2::from_diag(&ndarray::array![1., 2., 1.])
        );
    }

    #[test]
    fn test_adjacency() {
        let a_mat = adjacency(&undirected_unweighted_line_graph());
        assert_eq!(
            a_mat,
            &ndarray::array![[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]]
        );
    }

    #[test]
    fn test_incidence() {
        let c_mat = incidence(&undirected_unweighted_line_graph());
        assert_eq!(c_mat, ndarray::array![[-1., 0.], [1., -1.], [0., 1.]]);
    }

    #[test]
    fn test_laplacian() {
        let l_mat = laplacian(&undirected_unweighted_line_graph());
        assert_eq!(
            l_mat,
            ndarray::array![[1., -1., 0.], [-1., 2., -1.], [0., -1., 1.]]
        );
    }

    #[test]
    fn test_edge_weight() {
        let w_mat = edge_weight(&undirected_unweighted_line_graph());
        assert_eq!(w_mat, ndarray::Array2::eye(2));
    }
}
