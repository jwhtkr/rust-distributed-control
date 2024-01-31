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

pub mod mas;
pub mod graphs;
pub mod dynamics;
pub mod integrator;
pub use mas::*;
