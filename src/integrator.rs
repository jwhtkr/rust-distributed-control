//! Define integrators for dynamic systems. E.g., Euler, RK45, etc.

use std::{cmp::min_by, ops::Mul};

use ndarray::{Array1, Array2, LinalgScalar, ScalarOperand, ShapeBuilder};
use ndarray_linalg::Scalar;

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
    fn simulate(times: &Vec<T>, x0: &Array1<T>, dynamics: &D, input: &U) -> Array2<T> {
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
        history
    }
}

/// Implement Euler integration
///
/// I.e.,
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
        x0 + dynamics.dynamics(t0, x0, &u) * delta_t
    }
}

/// Implement the classic Runge-Kutta Order 4 integrator
///
/// I.e.,
/// $$
///     \int_{t_0}^{t_f} \dot{x} = x_0 + h / 6 (k_1 + 2k_2 + 2k_3 + k4)
/// $$
/// $$
///     k_1 = f(t_0, x_0, u(t_0, x_0))
/// $$
/// $$
///     k_2 = f(t_0 + h/2, x_0 + k1 h / 2, u(t_0 + h/2, x_0 + k1h/2))
/// $$
/// $$
///     k_3 = f(t_0 + h/2, x_0 + k2 h/2, u(t_0 + h/2, x_0 + k1h/2))
/// $$
/// $$
///     k_4 = f(t_0 + h, x_0 + k3h, u(t_0 + h, x_0 + k3h))
/// $$
/// where $h = t_f - t_0$.
pub struct RK4;

impl<T, D, U> Integrator<T, D, U> for RK4
where
    T: LinalgScalar + ScalarOperand + From<f64>,
    D: Dynamics<T>,
    U: Fn(T, &Array1<T>) -> Array1<T>,
{
    fn step(t0: T, tf: T, x0: &Array1<T>, dynamics: &D, input: &U) -> Array1<T> {
        let delta_t = tf - t0;
        let half = T::from(0.5);
        let t_half = delta_t * half;
        let k1 = &dynamics.dynamics(t0, x0, &input(t0, x0));
        let t_mid = t0 + t_half;
        let x_mid_1 = x0 + &k1.mul(t_half);
        let k2 = &dynamics.dynamics(t_mid, &x_mid_1, &input(t_mid, &x_mid_1));
        let x_mid_2 = x0 + &k2.mul(t_half);
        let k3 = &dynamics.dynamics(t_mid, &x_mid_2, &input(t_mid, &x_mid_2));
        let x_end_pre = x0 + &k3.mul(delta_t);
        let k4 = &dynamics.dynamics(tf, &x_end_pre, &input(tf, &x_end_pre));
        x0 + (k1 + k2.mul(T::from(2.0)) + k3.mul(T::from(2.0)) + k4)
            .mul(delta_t.mul(T::from(1. / 6.)))
    }
}

/// An order two adaptive step size Runge-Kutta method
///
/// I.e., A combination of Euler integration and Heun's method as in the
/// Wikipedia article on
/// [adaptive Runge-Kutta methods](https://en.wikipedia.org/wiki/Rung%E2%80%93Kutta_methods#Adaptive_Runge%E2%80%93Kutta_methods).
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveRK2 {
    tol: f64,
}

impl AdaptiveRK2 {
    pub fn new(tol: f64) -> Self {
        AdaptiveRK2 { tol }
    }

    pub fn adaptive_step<T, D, U>(
        &self,
        t0: T,
        tf: T,
        x0: &Array1<T>,
        dynamics: &D,
        input: &U,
    ) -> Array1<T>
    where
        T: LinalgScalar + Scalar + ScalarOperand + PartialOrd,
        D: Dynamics<T>,
        U: Fn(T, &Array1<T>) -> Array1<T>,
    {
        // Define the initial starting points and starting step
        let mut t_curr = t0;
        let mut x_curr = x0.clone();
        let mut step = T::from(0.5 * self.tol.powf(0.5)).unwrap();
        if step > (tf - t0) {
            // Make sure we don't over-step the bounds with the initial step
            // size (not likely, but just in case).
            step = tf - t0;
        }
        let mut i = 0;
        let mut j = 0;

        while t_curr < tf {
            // Detect if the step size has dropped too low.
            // TODO: Return Error instead of just panicing.
            if t_curr + step == t_curr {
                panic!("The step size dropped below machine precision.");
            }

            // Calculate the two stages needed for Euler/Huen methods.
            let k1 = &dynamics.dynamics(t_curr, &x_curr, &input(t_curr, &x_curr));
            let x_tmp = &x_curr + k1 * step;
            let k2 = &dynamics.dynamics(t_curr + step, &x_tmp, &input(t_curr + step, &x_tmp));

            // Calculate the error based on the difference between the two
            // estimates of the function.
            let err = (k2 - k1) * step * T::from(0.5).unwrap();
            let err_inf_norm: T = T::from_real(
                err.iter()
                    .map(|v| v.abs())
                    .max_by(|&a, &b| a.partial_cmp(&b).unwrap())
                    .unwrap(),
            );
            let x_inf_norm = T::from_real(
                x_curr
                    .iter()
                    .map(|v| v.abs())
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap(),
            );
            let max_err = T::from(self.tol).unwrap() * (T::one() + x_inf_norm);
            if err_inf_norm < max_err {
                // If the error is acceptable, advance by one step.
                t_curr += step;
                x_curr = &x_curr + (k1 + k2) * step * T::from(0.5).unwrap();
                j += 1;
            }
            // Calculate a "near" optimal step size multiplier (conservative due
            // to multiplication by 0.9)
            let mut multiplier = T::from(0.9).unwrap() * (max_err / err_inf_norm).powf(T::from(0.5).unwrap().re());
            // Don't allow the step size to grow too much.
            multiplier = min_by(multiplier, T::from(4.0).unwrap(), |a, b| {
                a.partial_cmp(b).unwrap()
            });
            // Ensure we don't go past the end time.
            step = min_by(multiplier * step, tf - t_curr, |a, b| {
                a.partial_cmp(b).unwrap()
            });
            i += 1;
        }
        println!("Adaptive step used {i} iterations with {j} steps");
        x_curr.into_owned()
    }
}

impl Default for AdaptiveRK2 {
    fn default() -> Self {
        AdaptiveRK2 { tol: 1e-8 }
    }
}

impl<T, D, U> Integrator<T, D, U> for AdaptiveRK2
where
    T: LinalgScalar + ScalarOperand + ndarray_linalg::Scalar + std::cmp::PartialOrd,
    D: Dynamics<T>,
    U: Fn(T, &Array1<T>) -> Array1<T>,
{
    fn step(t0: T, tf: T, x0: &Array1<T>, dynamics: &D, input: &U) -> Array1<T> {
        let integrator = AdaptiveRK2::default();
        integrator.adaptive_step(t0, tf, x0, dynamics, input)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, s, Array};

    use super::*;
    use crate::dynamics::LtiDynamics;

    #[test]
    fn test_adaptive_step_harmonic_oscillator() {
        let lti = LtiDynamics::new(array![[0., 1.], [-1., 0.]], array![[0.], [1.]]);
        let control = |_t, _x: &_| array![0.];
        let x_0 = array![2., 0.];
        let adapt_int = AdaptiveRK2::default();
        let start = std::time::Instant::now();
        let step = adapt_int.adaptive_step(0., 1.0, &x_0, &lti, &control);
        let duration = start.elapsed().as_secs_f64();
        println!("Step time: {duration:.5}");

        assert!(step.abs_diff_eq(&array![1.080604608324292, -1.682941965891026], 1e-8));
    }

    #[test]
    fn test_adaptive_step_stable() {
        let lti = LtiDynamics::new(array![[-1., 1.], [0., -1.]], array![[0.], [1.]]);
        let control = |_t, _x: &_| array![0.];
        let x_0 = array![2., 0.];
        let adapt_int = AdaptiveRK2::default();
        let start = std::time::Instant::now();
        let step = adapt_int.adaptive_step(0., 1.0, &x_0, &lti, &control);
        let duration = start.elapsed().as_secs_f64();
        println!("Step time: {duration:.5}");

        assert!(step.abs_diff_eq(&array![0.735758884760948, 0.], 1e-8));
    }

    #[test]
    fn test_adaptive_step_unstable() {
        let lti = LtiDynamics::new(array![[1., 1.], [0., 2.]], array![[0.], [1.]]);
        let control = |_t, _x: &_| array![0.];
        let x_0 = array![2., 0.];
        let adapt_int = AdaptiveRK2::default();
        let start = std::time::Instant::now();
        let step = adapt_int.adaptive_step(0., 1.0, &x_0, &lti, &control);
        let duration = start.elapsed().as_secs_f64();
        println!("Step time: {duration:.5}");

        assert!(step.abs_diff_eq(&array![5.436563669594182, 0.], 1e-7));
    }

    #[test]
    fn test_rk4_harmonic_oscillator() {
        let lti = LtiDynamics::new(array![[0., 1.], [-1., 0.]], array![[0.], [1.]]);
        let control = |_t, _x: &_| array![0.];
        let x_0 = array![2., 0.];
        let start = std::time::Instant::now();
        let times = Array::linspace(0., 1., 100).into_iter().collect();
        let states = RK4::simulate(&times, &x_0, &lti, &control);
        let duration = start.elapsed().as_secs_f64();
        println!("Step time: {duration:.5}");

        assert!(states.slice(s![.., -1]).abs_diff_eq(&array![1.080604608324292, -1.682941965891026], 1e-8));
    }

    #[test]
    fn test_rk4_stable() {
        let lti = LtiDynamics::new(array![[-1., 1.], [0., -1.]], array![[0.], [1.]]);
        let control = |_t, _x: &_| array![0.];
        let x_0 = array![2., 0.];
        let start = std::time::Instant::now();
        let times = Array::linspace(0., 1., 100).into_iter().collect();
        let states = RK4::simulate(&times, &x_0, &lti, &control);
        let duration = start.elapsed().as_secs_f64();
        println!("Step time: {duration:.5}");

        assert!(states.slice(s![.., -1]).abs_diff_eq(&array![0.735758884760948, 0.], 1e-8));
    }

    #[test]
    fn test_rk4_unstable() {
        let lti = LtiDynamics::new(array![[1., 1.], [0., 2.]], array![[0.], [1.]]);
        let control = |_t, _x: &_| array![0.];
        let x_0 = array![2., 0.];
        let start = std::time::Instant::now();
        let times = Array::linspace(0., 1., 100).into_iter().collect();
        let states = RK4::simulate(&times, &x_0, &lti, &control);
        let duration = start.elapsed().as_secs_f64();
        println!("Step time: {duration:.5}");

        assert!(states.slice(s![.., -1]).abs_diff_eq(&array![5.436563669594182, 0.], 1e-7));
    }
}
