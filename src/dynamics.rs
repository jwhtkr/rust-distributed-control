//! Define the various types of MAS dynamics.

use ndarray::{s, Array1, Array2, LinalgScalar};

/// Continuous-time dynamics of the form $\dot{x} = f(t, x, u)$.
/// I.e., fully non-linear and time-varying dynamics.
///
/// It is defined this way to allow full flexibility for
/// downstream users to implement any dynamics that are desired,
/// but this crate focuses on Linear time-invariant dynamics.
pub trait Dynamics<T: LinalgScalar> {
    /// Calculate the dynamics, i.e., $\dot{x} = f(t, x, u(t, x))$
    fn dynamics(self: &Self, t: T, x: &Array1<T>, u: &Array1<T>) -> Array1<T>;
    /// Get the dimension of the state $x$.
    fn n_state(self: &Self) -> usize;
    /// Get the dimension of the input $u$.
    fn n_input(self: &Self) -> usize;
}

/// Multi-agent system dynamics with possibly heterogeneous
/// dynamics for each agent.
///
/// This allows implementing this trait
/// as desired for downstream users, while this crate focuses on
/// homogeneous multi-agent systems (for now).
pub trait MasDynamics<T: LinalgScalar>: Dynamics<T> {
    /// Get the dynamics object for the $i^th$ agent (starting at index 0)
    ///
    /// Returns an error if the index is invalid ($i$ >= `self.n_agents()`).
    fn mas_dynamics(&self, i: usize) -> Result<&dyn Dynamics<T>, &str>;

    /// The number of agents in the multi-agent system.
    fn n_agents(&self) -> usize;

}

/// Provide a conversion from agent-wise to compact dynamics. I.e., the
/// state and input vectors (`x` and `u`) are comprised of the state and
/// input of each agent stacked/concatenated.
pub fn compact_dynamics<T: LinalgScalar>(mas_dynamics: &dyn MasDynamics<T>, t: T, x: &Array1<T>, u: &Array1<T>) -> Array1<T> {
    let all_dynamics: Vec<_> = (0..mas_dynamics.n_agents())
        .map(|i| mas_dynamics.mas_dynamics(i).unwrap())
        .collect();
    let n_x_total = all_dynamics.iter().fold(0, |acc, &el| acc + el.n_state());

    let mut x_next = Array1::zeros((n_x_total,));
    let mut x_i_start = 0;
    let mut u_i_start = 0;
    for i in 0..mas_dynamics.n_agents() {
        let _dyn = all_dynamics[i];
        let n_x_i = _dyn.n_state();
        let n_u_i = _dyn.n_input();
        let x_i_end = x_i_start + n_x_i;
        let u_i_end = u_i_start + n_u_i;

        let x_i = x.slice(s![x_i_start..x_i_end]).into_owned();
        let u_i = u.slice(s![u_i_start..u_i_end]).into_owned();

        x_next
            .slice_mut(s![x_i_start..x_i_end])
            .assign(&_dyn.dynamics(t, &x_i, &u_i));

        x_i_start = x_i_end;
        u_i_start = u_i_end;
    }

    return x_next;
}

/// Implement linear, time-invariant (LTI) dynamics. I.e.,
/// $\dot{x} = A x + B u$.
///
/// Example
/// ```
/// use ndarray::array;
/// use distributed_control::dynamics::{Dynamics, LtiDynamics};
///
/// let dynamics = LtiDynamics::new(array![[0., 1.], [0., 0.]], array![[0.], [1.]]);
/// assert_eq!(dynamics.dynamics(0., &array![1., 2.], &array![3.]), array![2., 3.]);
/// assert_eq!(dynamics.n_input(), 1);
/// assert_eq!(dynamics.n_state(), 2);
/// ```
#[derive(Debug)]
pub struct LtiDynamics<T: LinalgScalar> {
    a_mat: Array2<T>,
    b_mat: Array2<T>,
}

impl<T: LinalgScalar> LtiDynamics<T> {
    /// Create an LTI system from an $A$ matrix (`a_mat`) and a $B$ matrix (`b_mat`)
    pub fn new(a_mat: Array2<T>, b_mat: Array2<T>) -> LtiDynamics<T> {
        LtiDynamics { a_mat, b_mat }
    }
}

impl<T: LinalgScalar> Dynamics<T> for LtiDynamics<T> {
    fn n_input(self: &Self) -> usize {
        self.b_mat.ncols()
    }

    fn n_state(self: &Self) -> usize {
        self.a_mat.ncols()
    }

    fn dynamics(self: &Self, _t: T, x: &Array1<T>, u: &Array1<T>) -> Array1<T> {
        self.a_mat.dot(x) + self.b_mat.dot(u)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_linear_dynamics() {
        let _dyn = LtiDynamics::new(array![[0., 1.], [0., 0.]], array![[0.], [1.]]);
        assert_eq!(_dyn.n_input(), 1);
        assert_eq!(_dyn.n_state(), 2);
        assert_eq!(
            _dyn.dynamics(0., &array![1., 1.], &array![2.]),
            array![1., 2.]
        );
    }
}
