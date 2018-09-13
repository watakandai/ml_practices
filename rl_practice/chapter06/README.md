# RandomWalk

## Estimated Value
![Value](https://user-images.githubusercontent.com/11141442/45475204-9b0acb80-b776-11e8-83cc-7cde9095fab8.jpg)


## RMS ERROR
![RMS](https://user-images.githubusercontent.com/11141442/45475194-96461780-b776-11e8-9f64-053782cf8553.jpg)


# Monte Carlo Update
$$$
V(S_t) = V(S_t) + ¥alpha*(G_t - V(S_t))
$$$

# Temporal Difference Update
$$$
V(S_t) = V(S_t) + ¥alpha*(R_{t+1} + ¥gamma*V(S_{t+1}) - V(S_t))
$$$
