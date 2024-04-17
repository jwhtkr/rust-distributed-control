//! Define the various types of MAS dynamics.

use ndarray::{s, Array1, Array2, Axis, LinalgScalar};

/// Continuous-time dynamics of the form $\dot{x} = f(t, x, u)$.
/// I.e., fully non-linear and time-varying dynamics.
///
/// It is defined this way to allow full flexibility for
/// downstream users to implement any dynamics that are desired,
/// but this crate focuses on Linear time-invariant dynamics.
pub trait Dynamics<T: LinalgScalar> {
    /// Calculate the dynamics, i.e., $\dot{x} = f(t, x, u(t, x))$
    fn dynamics(&self, t: T, x: &Array1<T>, u: &Array1<T>) -> Array1<T>;
    /// Calculate the output, i.e., $y = g(t, x, u(t, x))$
    fn output(&self, t: T, x: &Array1<T>, u: &Array1<T>) -> Array1<T>;
    /// Get the dimension of the state $x$.
    fn n_state(&self) -> usize;
    /// Get the dimension of the input $u$.
    fn n_input(&self) -> usize;
    /// Get the dimension of the output $y$.
    fn n_output(&self) -> usize;
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
pub fn compact_dynamics<T: LinalgScalar>(
    mas_dynamics: &dyn MasDynamics<T>,
    t: T,
    x: &Array1<T>,
    u: &Array1<T>,
) -> Array1<T> {
    let all_dynamics: Vec<_> = (0..mas_dynamics.n_agents())
        .map(|i| mas_dynamics.mas_dynamics(i).unwrap())
        .collect();

    let x_nexts: Vec<_> = all_dynamics
        .iter()
        .scan((0, 0), |(i_x, i_u), &d| {
            let x_i = x.slice(s![*i_x..*i_x + d.n_state()]).to_owned();
            let u_i = u.slice(s![*i_u..*i_u + d.n_input()]).to_owned();
            *i_x += d.n_state();
            *i_u += d.n_input();
            Some(d.dynamics(t, &x_i, &u_i))
        })
        .collect();
    let x_nexts_view: Vec<_> = x_nexts.iter().map(|v| v.view()).collect();
    ndarray::concatenate(Axis(0), x_nexts_view.as_slice()).unwrap()
}

/// Provide a conversion from agent-wise to compact output.
pub fn compact_output<T: LinalgScalar>(
    mas_dynamics: &dyn MasDynamics<T>,
    t: T,
    x: &Array1<T>,
    u: &Array1<T>,
) -> Array1<T> {
    let all_dyn: Vec<_> = (0..mas_dynamics.n_agents())
        .map(|i| mas_dynamics.mas_dynamics(i).unwrap())
        .collect();

    let outputs: Vec<_> = all_dyn
        .iter()
        .scan((0, 0), |(i_x, i_u), &d| {
            let x_i = x.slice(s![*i_x..*i_x + d.n_state()]).to_owned();
            let u_i = u.slice(s![*i_u..*i_u + d.n_input()]).to_owned();
            *i_x += d.n_state();
            *i_u += d.n_input();
            Some(d.output(t, &x_i, &u_i))
        })
        .collect();
    let outputs_views: Vec<_> = outputs.iter().map(|v| v.view()).collect();
    ndarray::concatenate(Axis(0), outputs_views.as_slice()).unwrap()
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
#[derive(Debug, Clone)]
pub struct LtiDynamics<T: LinalgScalar> {
    pub a_mat: Array2<T>,
    pub b_mat: Array2<T>,
    pub c_mat: Array2<T>,
    pub d_mat: Array2<T>,
}

impl<T: LinalgScalar> LtiDynamics<T> {
    /// Create an LTI system from an $A$ matrix (`a_mat`) and a $B$ matrix (`b_mat`)
    pub fn new(a_mat: Array2<T>, b_mat: Array2<T>) -> LtiDynamics<T> {
        let n_state = a_mat.ncols();
        let n_input = b_mat.ncols();
        LtiDynamics {
            a_mat,
            b_mat,
            c_mat: Array2::zeros((0, n_state)),
            d_mat: Array2::zeros((0, n_input)),
        }
    }
}

impl<T: LinalgScalar> Dynamics<T> for LtiDynamics<T> {
    fn n_input(&self) -> usize {
        self.b_mat.ncols()
    }

    fn n_state(&self) -> usize {
        self.a_mat.ncols()
    }

    fn n_output(&self) -> usize {
        self.c_mat.nrows()
    }

    fn dynamics(&self, _t: T, x: &Array1<T>, u: &Array1<T>) -> Array1<T> {
        self.a_mat.dot(x) + self.b_mat.dot(u)
    }

    fn output(&self, _t: T, x: &Array1<T>, u: &Array1<T>) -> Array1<T> {
        self.c_mat.dot(x) + self.d_mat.dot(u)
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
