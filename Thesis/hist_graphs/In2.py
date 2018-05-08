Xs = np.arange(1, 10.5, 0.5)
np.random.seed(1345)
L = len(Xs)
A1 = 1.5/(Xs+0.5)+np.random.normal(0, .02, L)
A2 = 1/np.sqrt(Xs+2)+np.random.normal(0, .02, L)
A3 = np.maximum(1.2-0.16*Xs, 0.1)+np.random.normal(0, .02, L)

M = np.array([A1,A2,A3])
A4 = np.min(M, axis=0)+abs(np.random.normal(0, .02, L))
M = np.array([A1,A2,A3,A4])

fig, ax = plt.subplots()
ax.plot(Xs, M.T)
ax.set_ylabel('Error')
ax.set_xlabel('epsilon')

regrets = np.divide(M, np.min(M, axis=0)).mean(axis=1)

plt.legend(['A1', 'A2', 'A3', 'Tool'])
plt.savefig('Perf', bbox_inches='tight')
def plot_regrets(regrets):
    fig, ax = plt.subplots()
    plt.bar(ind, regrets, width)
    ax.set_ylabel('Regret')  
    ax.set_xticks(ind)
    ax.set_xticklabels(['A1', 'A2', 'A3', 'Tool'])
    plt.savefig('regret', bbox_inches='tight')
plot_regrets(regrets)
