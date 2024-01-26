//! This module contains the primary structs, types, and traits
//! for use in defining and analyzing a MAS.


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mas_creation() {
        use petgraph;
        let g = petgraph::graph::UnGraph::<i32, ()>::from_edges(&[(0,1), (1,2), (2,3)]);
    }
}
