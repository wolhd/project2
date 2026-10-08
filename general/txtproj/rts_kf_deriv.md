In [Jeffrey W. Miller's course materials](https://jwmi.github.io/ASM/6-KalmanFilter.pdf), the mathematical link becomes clear when analyzing how the HMM backward variable transitions into the Rauch–Tung–Striebel (RTS) equations. [1] 
To see exactly where the HMM backward variable transforms into the Kalman framework, we must break down the transition step.
------------------------------
## Step 1: The Continuous HMM Backward Equation
As seen in standard HMM notation, the backward variable $r_j(z_j) = p(x_{j+1:n} \mid z_j)$ operates recursively from the end of the sequence. In a continuous-state model (like the Linear-Gaussian state-space model used in Kalman setups), the discrete summation becomes an integral: [1] 
$$r_j(z_j) = \int p(x_{j+1:n} \mid z_{j+1}) \, p(z_{j+1} \mid z_j) \, dz_{j+1}$$ 
By splitting the future observations $x_{j+1:n}$ into the immediate next observation $x_{j+1}$ and the remaining future $x_{j+2:n}$, the integral expands to:
$$r_j(z_j) = \int \underbrace{p(x_{j+2:n} \mid z_{j+1})}_{r_{j+1}(z_{j+1})} \, \underbrace{p(x_{j+1} \mid z_{j+1})}_{\text{Emission}} \, \underbrace{p(z_{j+1} \mid z_j)}_{\text{Transition}} \, dz_{j+1}$$ 
## Step 2: Injecting the "Magic of Gaussians"
The exact moment this transitions into the RTS Kalman Smoother occurs when Miller applies the multivariate normal parameters of the model: [1] 

   1. Transition model: $z_{j+1} = A z_j + w_j$, where noise $w_j \sim \mathcal{N}(0, Q)$. Therefore, $p(z_{j+1} \mid z_j) = \mathcal{N}(z_{j+1}; A z_j, Q)$.
   2. Emission model: $x_{j+1} = H z_{j+1} + v_{j+1}$, where noise $v_{j+1} \sim \mathcal{N}(0, R)$. Therefore, $p(x_{j+1} \mid z_{j+1}) = \mathcal{N}(x_{j+1}; H z_{j+1}, R)$.

Because multiplying and integrating Gaussians always results in another Gaussian, the backward variable $r_j(z_j)$ is mathematically proven to be a quadratic function in log-space.
Instead of tracking an infinite number of continuous probability points, the backward pass only needs to track two parameters at each step $j$:

* 
* A vector $\hat{z}_j^b$ (the backward mean estimate).
* A matrix $P_j^b$ (the backward covariance/information matrix).
* 

## Step 3: The Transition to RTS Variables
While you can compute $r_j(z_j)$ strictly backward using those two variables (often called the Two-Filter Smoother), the standard RTS Smoother uses a algebraic shortcut.
Instead of computing $r_j(z_j)$ completely independently, the RTS approach combines the current information with the results from the forward Kalman Filter pass ($s_j(z_j) = p(z_j \mid x_{1:j})$) at the same time.
When evaluating the total smoothed distribution $p(z_j \mid x_{1:n}) \propto s_j(z_j) r_j(z_j)$, the product of the forward Gaussian and backward Gaussian triggers a matrix identity. This identity bypasses the need to explicitly calculate the raw backward likelihood $r_j(z_j)$ altogether.
Instead, it calculates the Smoothed Gain Matrix ($G_j$):
$$G_j = P_j F^T (P_{j+1}^-)^{-1}$$ 
Where:

* 
* $P_j$ is the forward filtered covariance at time $j$.
* $F$ is the state transition matrix.
* $P_{j+1}^-$ is the predicted forward covariance for the next step.
* 

This gain matrix is then used to update the final smoothed mean vector $\hat{z}_{j \mid n}$ backward:
$$\hat{z}_{j \mid n} = \hat{z}_{j \mid j} + G_j (\hat{z}_{j+1 \mid n} - \hat{z}_{j+1 \mid j})$$ 
## Summary
The HMM backward algorithm looks like an intimidating integration problem over continuous space. However, because everything is Gaussian, Miller demonstrates that the integral simplifies into linear algebra. The RTS smoother updates are merely the HMM backward algorithm optimized so that you only pass the difference between your forward predictions and your actual backward knowledge. [1] 
Would you like to walk through the algebraic completion of the squares that proves the product of the HMM forward-backward variables yields a Gaussian, or look at a specific equation from Miller's PDF that is giving you trouble?

[1] [https://jwmi.github.io](https://jwmi.github.io/ASM/6-KalmanFilter.pdf)
