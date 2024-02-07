//! A library for performing and analyzing distributed control
//!
//! The primary uses involve creating a Multi-agent System (MAS),
//! inspecting and analyzing the MAS, and then simulating the MAS
//! through time.
//!
//! The mas module defines the basic MAS structs, types, and traits
//! with other modules supporting this primary module. However, the
//! most commonly used functionality is re-exported to the top level
//! for ease-of-use.
//!
//! Examples:
//! ```
//! use distributed_control as dc;
//! use dc::integrator::Integrator;
//! use ndarray::{array, Array1};
//!
//! let lin_dyn = dc::LtiDynamics::new(array![[0.]], array![[1.]]);
//! let mas = dc::HomMas::new(&lin_dyn, 3);
//! let initial_states = array![-1., 0., 1.];
//! let laplacian = array![[1., -1., 0.], [-1., 2., -1.], [0., -1., 1.]];
//! let u = |_t: f64, x: &Array1<f64>| -laplacian.dot(x);
//!
//! let states = dc::EulerIntegration::simulate(
//!     (0..100).map(|i| i as f64 * 0.1).collect(),
//!     &initial_states,
//!     &mas,
//!     &u,
//! );
//! println!("{states}");
//! println!("{}", states.column(states.ncols() - 1));
//! ```

pub mod mas;
pub mod graphs;
pub mod dynamics;
pub mod integrator;
pub mod control_laws;
pub use mas::*;
pub use dynamics::LtiDynamics;
pub use integrator::EulerIntegration;
