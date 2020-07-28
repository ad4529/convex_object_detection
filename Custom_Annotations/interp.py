from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

new_dict = np.load('Train_hulls_cent_persons.npy', allow_pickle=True)
new_ids = list(new_dict[()].keys())
idx = np.random.randint(0, len(new_ids))
new_anns = new_dict[()][new_ids[idx]][0]
for ann in new_anns:
    vertices = ann.reshape(8,2)
    f1 = interp1d(vertices[:2,0], vertices[:2,1], kind='linear')
    new_x = np.linspace(vertices[0,0], vertices[1,0], 3)
    plt.plot(vertices[:2,0], vertices[:2,1], 'b', new_x, f1(new_x), 'go')
    plt.show()
    exit(1)
# f1 = interp1d(x, y, kind='linear')
# print(x.shape)
# print(f1(x).shape)
