# MPC simulation of catching a nosiy prey

This uses the do-mpc package to simulate an agent catching a prey. 

*One agent* section creates one agent and makes an animation of one episode. The agent has a simple linear model of prey's movement and the goal of minimizing the distance to it. You can add noise to prey's movement and change prediction horizon and how frequently the optimal action is computed. 

*Many agents* section simulates agents with different prediction intervals (how often you compute the optimal action) at different levels of noise in prey's movement. It plots catch times depending on those two variables. 
