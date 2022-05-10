import gym

import numpy as np

from typing import List, Tuple

def discretize(state) -> Tuple[int, int, int, int]:
    """
    Discretize the state space as follows:
    - Cart position: [-2.4, 2.4] -> {0, 1, 2}, where
        0: -2.4...-0.8
        1: -0.8...0.8
        2: 0.8...2.4
    - Cart velocity: [-Inf, Inf] -> {0, 1, 2}, where
        0: -Inf...-0.5
        1: -0.5...0.5
        2: 0.5...Inf
    - Pole angle: [-.2095, .2095] -> {0, 1, 2, 3, 4, 5}, where
        0: -0.2095...-0.1047
        1: -0.1047...-0.01745
        2: -0.01745...0
        3: 0...0.01745
        4: 0.01745...0.1047
        5: 0.1047...0.2095
    - Pole velocity: [-Inf, Inf] -> {0, 1, 2}, where
        0: -Inf...-0.872
        1: -0.872...0.872
        2: 0.872...Inf
    """
    cart_pos, cart_vel, pole_ang, pole_vel = state

    if -2.4 <= cart_pos and cart_pos < -0.8:
        cart_pos = 0
    elif -0.8 <= cart_pos and cart_pos < 0.8:
        cart_pos = 1
    elif 0.8 <= cart_pos and cart_pos <= 2.4:
        cart_pos = 2

    assert 0 <= cart_pos <= 2, f"Cart position {cart_pos} is out of bounds."

    if cart_vel < -0.5:
        cart_vel = 0
    elif cart_vel <= 0.5:
        cart_vel = 1
    else:
        cart_vel = 2

    if -0.2095 <= pole_ang and pole_ang < -0.1047:
        pole_ang = 0
    elif -0.1047 <= pole_ang and pole_ang < -0.01745:
        pole_ang = 1
    elif -0.01745 <= pole_ang and pole_ang < 0:
        pole_ang = 2
    elif 0 <= pole_ang and pole_ang < 0.01745:
        pole_ang = 3
    elif 0.01745 <= pole_ang and pole_ang < 0.1047:
        pole_ang = 4
    elif 0.1047 <= pole_ang and pole_ang <= 0.2095:
        pole_ang = 5

    assert pole_ang in range(6), f"Pole angle {pole_ang} is out of bounds."

    if pole_vel < -0.872:
        pole_vel = 0
    elif pole_vel <= 0.872:
        pole_vel = 1
    else:
        pole_vel = 2

    return cart_pos, cart_vel, pole_ang, pole_vel


def flatten_discrete_state(state: Tuple[int, int, int, int]) -> np.array:
    """Split the state vector into a region binary vector of size 162 (3*3*6*3)."""
    cart_pos, cart_vel, pole_ang, pole_vel = state
    result = [0] * 162
    result[cart_pos * (3*6*3) + cart_vel * (3*6) + pole_ang * 3 + pole_vel] = 1
    return np.array(result)


def activation(z: float) -> float:
    """Return +1 (right) if z is greater than or equal to 0, -1 (left) otherwise."""
    return 1 if z >= 0 else -1


def activation_to_action(activation: float) -> int:
    """Return the action corresponding to the given activation."""
    return 0 if activation < 0 else 1


def noise() -> float:
    """Return a random number from a Gaussian distribution with mean 0 and stdev 0.1."""
    return np.random.normal(0, 0.1)


def main():
    env = gym.make('CartPole-v0')

    UPDATE_RATE = 0.2
    TRACE_DECAY_RATE = 0.5

    for e in range(20):
        t = 0
        weights = np.repeat(0, 162)
        eligibility = np.repeat(0, 162)

        state = env.reset()
        while True:
            env.render()

            # observation
            state_discretized = discretize(state)
            print(f"Episode {e}: t = {t}: State = {state_discretized}")
            state_vector = flatten_discrete_state(state_discretized)

            # action
            output = activation(np.dot(weights, state_vector) + noise())
            action = activation_to_action(output)

            state, reward, done, info = env.step(action)

            if done:
                print(f"=== DONE ===")
                break

            # update
            weights = weights + UPDATE_RATE * reward * eligibility
            eligibility = TRACE_DECAY_RATE * eligibility + (1 - TRACE_DECAY_RATE) * output * state_vector

            t += 1

    env.close()

if __name__ == "__main__":
    main()