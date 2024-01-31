//! Define integrators for dynamic systems. E.g., Euler, RK45, etc.

use ndarray::{Array1, Array2, LinalgScalar, ScalarOperand, ShapeBuilder};

use crate::dynamics::Dynamics;

/// Define the interface for integrating dynamics.
pub trait Integrator<T, D, U>
where
    T: LinalgScalar,
    D: Dynamics<T>,
    U: Fn(T, &Array1<T>) -> Array1<T>,
{
    /// Integrate over the dynamics and input for one time step from
    /// `t0` to `tf`, with initial state `x0`.
    ///
    /// Integrates over the dynamics with the given
    /// input that is a function of time and state (ignore either or
    /// both if desired).
    /// It is expected that most of the time `simulate` will be called
    /// with the input function being a control law.
    /// However, sometimes it may be helpful to manually call `step`
    /// (e.g., the input function should be switched according to some
    /// criteria that is easier to check outside of the input function
    /// itself).
    fn step(t0: T, tf: T, x0: &Array1<T>, dynamics: &D, input: &U) -> Array1<T>;

    /// Simulate the dynamics with the input function over a time
    /// vector, `times`, starting at initial state `x0`.
    ///
    /// This integrates over the dynamics with the given input as a
    /// function of time and state (ignore either or both if desired).
    /// This allows user-defined adaptive steps if desired, although
    /// the expectation is to use a uniform sampling of points over
    /// the desired simulation window. The default implementation
    /// iteratively applies the `step` method over each adjacent pair
    /// of times in the `times` argument and collects the results in
    /// an array.
    fn simulate(times: Vec<T>, x0: &Array1<T>, dynamics: &D, input: &U) -> Array2<T> {
        let mut history = Array2::zeros((x0.len(), times.len()).f());
        history.column_mut(0).assign(x0);
        let mut x_curr = x0.clone();
        for (i, &t0) in times.iter().enumerate() {
            if i == times.len() - 1 {
                break;
            };
            let tf = times[i + 1];
            let x_next = Self::step(t0, tf, &x_curr, dynamics, input);
            history.column_mut(i + 1).assign(&x_next);
            x_curr = x_next;
        }
        return history;
    }
}

/// Implement Euler integration, i.e.,
/// $\int_{t_0}^{t_f} \dot{x} = x_0 + (t_f - t_0) f(t_0, x_0, u(t_0, x_0))$
pub struct EulerIntegration;

impl<T, D, U> Integrator<T, D, U> for EulerIntegration
where
    T: LinalgScalar + ScalarOperand,
    D: Dynamics<T>,
    U: Fn(T, &Array1<T>) -> Array1<T>,
{
    fn step(t0: T, tf: T, x0: &Array1<T>, dynamics: &D, input: &U) -> Array1<T> {
        let delta_t = tf - t0;
        let u = input(t0, x0);
        return x0 + dynamics.dynamics(t0, x0, &u) * delta_t;
    }
}
