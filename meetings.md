# Meeting 2025-05-07
- Went over Ivan's first pass implementation of Mujoco inverted pendulum task in gymnasium (Ivan/MPC_Mujoco_Cartpole/Cartpole_Control.ipynb)
- Added chat gpt's integration of Ivan's code with Dan's bird code
- Dan and Ivan will work over the weekend on getting this integrated code to run

# Meeting 2025-04-30
- Decided to implement RL/MPC in Mujoco environments in gymnasium so that we can easily train RL models or MPC models to solve different tasks
- The first task will be the inverted Mujoco pendulum in gymnasium

# Meeting 2025-04-16
- Went over the relation between SARSA and Dyna timesteps and n_planning
  - Should basically be interchangeable 
  - 1 Dyna timestep and 1000 planning steps should equal 1000 SARSA timesteps

- Fixed the dyna model (was not cycling actions, had action in place of next_action)
- Fixed the plotting errors which arose when changing the grid dims
- Talked about our goals for the comparisons between Dyna and MPC (see notes below)

 - Decided on next steps and to use github issues to track them:
 	- Finishing dyna model learning
 	- Implement continuum of shittiness in dyna
 	- Implement MPC

# Meeting 2025-04-09
Skipped, 

# Meeting 2025-04-02
- Went over wind and additional reward implementations
- Implemented Dyna

# Meeting 2025-03-26
1. Dan developed code for doing SARSA and eventually MPC in a 3D grid world where the bird moves in x-y plus can flap wings or not to move in y. The code currently has no wind, but should be easy to add.

2. Ivan developed code to compare deep Q learning to PPO to MPC with different horizons and frequencies of recomputing the optimal trajectory. This is implemented using gym and is on github:
Result plot: Ivan/MPC_RL_Cartpole/Results/average_episode_lengths.png
Code: Ivan/MPC_RL_Cartpole/Cartpole_Control.ipynb

3. We are still confused about where we are going with this, but we all think it's cool/interesting. Maybe it's worth briefly revisiting the paper Jatan presented a while back to check how they motivate MPC: https://ieeexplore.ieee.org/abstract/document/6386025

4. We discussed the potential most interesting questions:
a. Optimization of the horizon or recompute frequency
b. Dealing with noise or generalization to new environments

Next steps
1. Dan will work on adding wind and training the SARSA bird plus thinking about the rollout model [broadly what we discussed, but I might be missing smth]
2. Ivan will wrap up the cartpole simulations to do more detailed analysis of the planning horizon, recompute frequency, and noise in the system
3. When Dan has the first version of the SARSA bird in the wind env. Ivan can adjust the MPC code so that it can be plugged into the bird environment
@Dan & Atsushi please add/edit stuff, I think I butchered this quite a bit.
@Jatan: can you re-add Dan to the Github repo? The invitation worked, but expired.

The only things I would add to your summary:
- it would be interesting to address learning appropriate horizons, recomputes, etc conditional on environments or something
- it would probably be worth figuring out a few "new model desiderata" that we could address to structure our priorities and goals
   - ex: doing MPC over options, or learning state-abstraction
   - ex: dealing with different levels or sorts of surprise to determine recomputes / horizons / planning
   - ex: deciding when to refine model vs use shitty model
   - if we could define a few things like this we could design an environment requiring them, then show we can make a model that beats others

- Ivan: I further improved my code and pushed it on github. Here's a simulation with adding noise in the cartpole (always the same across the models in each episode).
- Ivan: If we add a bit of noise (in all 4 dims - angle, angle velocity, cart, cart velocity) not much changes. MPC with horizon 50 and recomputing every step maxes out, and so does PPO.


# Meeting Notes 2025-03-19

- Next steps:
  - Dan - Think/try to setup a basic model so we can train with RL (i.e., A & B matrices)
  - Ivan & Atsushi - We'll take a stab at creating a gym environment and make it easily installable
  - Jatan - Work on the 3d simulations with the goal of getting a very preliminary dataset we can work with to setup the rest

# Meeting Notes 2025-03-12
- Went over the soaring bird PNAS paper.
- One pipeline to investigate the impact of model misspecification on soaring performance:
    1. Take the optimal model by either doing optimal control from the equations or training an RL agent.
    2. Take the "shitty bird" which has a super bad model (A and B).
    3. Mix in different levels of shitty bird vs. optimal model to get a continuum of how bad your model is.
    4. This way we can look at what you and I were discussing last time - how far you can get with a bad-ish model if you keep recomputing what to do often.

- when comparing SARSA bird to Shitty bird we can look at how they perform on parameters that are outside the training set.
- hopefully that shitty bird does better.

- next time we'll start going over Jatan's code and thinking more how to actually implement this.

-  A few thoughts wrt comparing the SARSA bird to the Shitty bird: 
  - I think it's obvious that birds don't do what the SARSA bird does, and that's what could make this work interesting. 
  - SARSA bird knows what the high value action of every possible state in the training set is.
  - So the SARSA bird is a decent path towards an engineering solution for the small glider, but not a great way to understand birds.
  - Furthermore, it's a great solution only if you know exactly which combos of temperatures/pressures the little plan will experience so you can train on those parameter combos.

