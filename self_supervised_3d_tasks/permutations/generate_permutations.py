from pathlib import Path
import numpy as np


if __name__ == "__main__":
    perms = []

    for i in range(16):
        pp = np.random.permutation(16)
        # print(pp)

        for ref in perms:
            if (pp==ref).all():
                print("REGENERATE!!!!")

        perms.append(pp)

    perms = np.stack(perms)

    permutation_path = str('/Users/laurentletourneau-guillon/Dropbox/CHUM/RECHERCHE/2022 Eleyine self-supervised segmentations ICH/HealthML/self_supervised_3d_tasks/permutations/permutations3d_100_27.npy')

    print(permutation_path)

    with open(permutation_path, 'wb') as f:
        np.save(f, perms)
