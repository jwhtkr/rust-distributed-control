//! This module contains the primary structs, types, and traits
//! for use in defining and analyzing a Multi-agent System (MAS).

use ndarray::LinalgScalar;

use crate::dynamics::{compact_dynamics, Dynamics, MasDynamics};

/// A homogenous MAS, i.e., the dynamics of each agent are identical.
pub struct HomMas<'a, T: LinalgScalar> {
    hom_dynamics: &'a dyn Dynamics<T>,
    n_agents: usize,
}

impl<'a, T: LinalgScalar> HomMas<'a, T> {
    /// create a new homogenous MAS with the given dynamics and number of agents.
    pub fn new(hom_dynamics: &dyn Dynamics<T>, n_agents: usize) -> HomMas<T> {
        HomMas {
            hom_dynamics,
            n_agents,
        }
    }
}

impl<'a, T: LinalgScalar> MasDynamics<T> for HomMas<'a, T> {
    fn mas_dynamics(&self, i: usize) -> Result<&dyn Dynamics<T>, &str> {
        if i < self.n_agents {
            Ok(self.hom_dynamics)
        } else {
            Err("The agent index exceeds the number of agents.")
        }
    }
    fn n_agents(&self) -> usize {
        self.n_agents
    }
}

impl<'a, T: LinalgScalar> Dynamics<T> for HomMas<'a, T> {
    fn dynamics(
        &self,
        t: T,
        x: &ndarray::prelude::Array1<T>,
        u: &ndarray::prelude::Array1<T>,
    ) -> ndarray::prelude::Array1<T> {
        compact_dynamics(self, t, x, u)
    }

    fn n_input(&self) -> usize {
        (0..self.n_agents())
            .map(|i| self.mas_dynamics(i).unwrap().n_input()).sum()
    }

    fn n_state(&self) -> usize {
        (0..self.n_agents())
            .map(|i| self.mas_dynamics(i).unwrap().n_state()).sum()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;
    use crate::dynamics;

    #[test]
    fn test_hom_mas() {
        let lin_dyn = dynamics::LtiDynamics::new(array![[0.]], array![[1.]]);
        let mas = HomMas::new(&lin_dyn, 3);
        assert_eq!(mas.n_agents(), 3);
        assert_eq!(
            mas.mas_dynamics(1)
                .unwrap()
                .dynamics(0., &array![1.,], &array![2.]),
            array![2.]
        );
        assert_eq!(
            mas.dynamics(0., &array![1., 2., 3.], &array![1., 1., 1.]),
            array![1., 1., 1.]
        );
    }

    #[test]
    #[should_panic(expected = "agent index exceeds")]
    fn test_hom_mas_out_of_index() {
        let lin_dyn = dynamics::LtiDynamics::new(array![[0.]], array![[1.]]);
        let mas = HomMas::new(&lin_dyn, 3);
        mas.mas_dynamics(3).unwrap();
    }
}
