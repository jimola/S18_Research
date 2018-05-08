np.random.seed(12345)
eps = 0.2
I1 = np.random.uniform(0, 1, 16)
N1 = np.random.laplace(0, eps, 16)
A1 = I1+N1
while((A1 < 0).sum() > 0):
    N1[A1 < 0] = np.random.laplace(0, eps, (A1 < 0).sum())
    A1 = I1+N1

#I2 = np.random.uniform(0, 1, 16)
I2 = I1.copy()
I2[2] += 0.7
N2 = np.random.laplace(0, eps, 16)
A2 = I2+N2
while((A2 < 0).sum() > 0):
    N2[A2 < 0] = np.random.laplace(0, eps, (A2 < 0).sum())
    A2 = I2+N2

def get_plots(I, name):
    fig, ax = plt.subplots(frameon=False)
    ax.set_xticks(np.arange(0, 5)-0.5)
    ax.set_yticks(np.arange(0, 5)-0.5)
    ax.imshow(I.reshape(4,4), cmap='RdBu', vmin=-1, vmax=1)
    ax.grid(color='w', linestyle='-', linewidth=6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', bottom=False, left=False, labelleft=False)
    #with open(name + '.png', 'wb') as outfile:
    #    fig.canvas.print_png(outfile)
    plt.savefig(name, bbox_inches='tight')

get_plots(A1, 'A1')
get_plots(A2, 'A2')
get_plots(N1, 'N1')
get_plots(N2, 'N2')
get_plots(I1, 'I1')
get_plots(I2, 'I2')

I3 = I1.copy().reshape(4, 4)
I3[:2,:2] = I3[:2,:2].mean()
I3[:2,2:] = I3[:2,2:].mean()
I3[2:,:2] = I3[2:,:2].mean()
I3[2:,2:] = I3[2:,2:].mean()

I4 = I2.copy().reshape(4, 4)
I4[:2,:2] = I4[:2,:2].mean()
I4[:2,2:] = I4[:2,2:].mean()
I4[2:,:2] = I4[2:,:2].mean()
I4[2:,2:] = I4[2:,2:].mean()

get_plots(I3, 'I3')
get_plots(I4, 'I4')

I3[:2,:2] = I3[:2,:2]+np.random.laplace(0, eps) / 4
I3[:2,2:] = I3[:2,2:]+np.random.laplace(0, eps) / 4
I3[2:,:2] = I3[2:,:2]+np.random.laplace(0, eps) / 4
I3[2:,2:] = I3[2:,2:]+np.random.laplace(0, eps) / 4

I4[:2,:2] = I4[:2,:2]+np.random.laplace(0, eps) / 4
I4[:2,2:] = I4[:2,2:]+np.random.laplace(0, eps) / 4
I4[2:,:2] = I4[2:,:2]+np.random.laplace(0, eps) / 4
I4[2:,2:] = I4[2:,2:]+np.random.laplace(0, eps) / 4

get_plots(I3, 'A3')
get_plots(I4, 'A4')

It = np.array([[0.8, 0.75, 0.44, 0.21],[0.84, 0.67, 0.67, 0.55],
               [0.2, 0.25, 0.71, 0.55],[0.2, 0.5, 0.9, 1.0]])
Ot = It + np.random.laplace(0, eps, (4,4))
Ot2 = It.copy()

Ot2[:2,:2] = Ot2[:2,:2].mean()+np.random.laplace(0, eps) / 4
Ot2[:2,2:] = Ot2[:2,2:].mean()+np.random.laplace(0, eps) / 4
Ot2[2:,:2] = Ot2[2:,:2].mean()+np.random.laplace(0, eps) / 4
Ot2[2:,2:] = Ot2[2:,2:].mean()+np.random.laplace(0, eps) / 4
get_plots(It, 'It')
get_plots(Ot, 'Ot')
get_plots(Ot2, 'Ot2')
