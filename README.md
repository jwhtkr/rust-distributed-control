# rust-distributed-control
Model, analyze and simulate multi-agent, distributed dynamic systems.

## Overview
A multi-agent system is one where there is more than one "agent" interacting with the environment. While an agent often is a distinct entity, e.g., autonomous vehicle, robot, etc., an agent may also be any portion of a system, e.g., a sensor in a sensor network. A distributed system is one in which each agent only receives communication (either direct data transfer via, e.g., bluetooth, or sensing via, e.g., LIDAR) from a subset of the other agents, usually modeled by a communication graph.

The dynamics of the agents in a distributed multi-agent system can be either homogeneous, in which case each agent has identical dynamics, or heterogeneous. Heterogeneous systems can either have dynamics of equal dimension (both state and input dimensions), where the dynamics differ, but the state and inputs of each system are the same (including the same "meaning" assigned to each state or input entry) or can have dynamics that differ in state and input dimensions as well.

This library focuses on linear, time-invariant (LTI) dynamics, although it provides the tools to extend beyond this, including to non-linear and time-varying dynamics.

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

Any contributions via the submission of pull requests, filing issues, or other methods are welcome and encouraged!
