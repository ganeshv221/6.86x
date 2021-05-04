import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")
LL = np.zeros(5)
# TODO: Your code here
K = 3
seed = 1
# for seed in range(5):
#   mixture, post = common.init(X, K, seed)
#   mixture, post, cost = kmeans.run(X, mixture, post)
#   common.plot(X, mixture, post, seed)

# for seed in range(5):
#   mixture, post = common.init(X, K, seed)
#   mixture, post, LL[seed] = naive_em.run(X, mixture, post)
#   common.plot(X, mixture, post, seed)

# print(np.max(LL))
# print(LL)

mixture, post = common.init(X, K, seed)
# post, LL = naive_em.estep(X, mixture)
# mixture = naive_em.mstep(X, post)
mixture, post, LL = naive_em.run(X, mixture, post)
bic = common.bic(X, mixture, LL)

print(bic)

# mixture, post, LL = em.run(X, mixture, post)
# print(mixture.mu)
# print(mixture.var)
# print(mixture.p)
# print(post)
# print(LL)