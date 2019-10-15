import pickle
import matplotlib.pyplot as plt

with open('reward.pickle', mode='rb') as f:
    reward_store = pickle.load(f)
with open('max_x.pickle', mode='rb') as f:
    max_x_store = pickle.load(f)

plt.plot(reward_store)
plt.show()
plt.close("all")
plt.plot(max_x_store)
plt.show()
