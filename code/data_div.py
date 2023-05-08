import numpy as np
from torchvision import datasets, transforms


def iid_mnist(dataset, n_client):
    items = int(len(dataset) / n_client)
    dict_cl = {}
    idx = [i for i in range(len(dataset))]
    for i in range(n_client):
        dict_cl[i] = set(np.random.choice(idx, items, replace=False))

        idx = list(set(idx) - dict_cl[i])
    return dict_cl


def iid_fmnist(dataset, n_client):
    items = int(len(dataset) / n_client)
    dict_cl, idx = {}, [i for i in range(len(dataset))]
    for i in range(n_client):
        dict_cl[i] = set(np.random.choice(idx, items, replace=False))

        idx = list(set(idx) - dict_cl[i])
    # print(dict_cl)
    return dict_cl


def iid_cifar(dataset, n_client):
    items = int(len(dataset) / n_client)
    dict_cl, idx = {}, [i for i in range(len(dataset))]
    for i in range(n_client):
        dict_cl[i] = set(np.random.choice(idx, items, replace=False))

        idx = list(set(idx) - dict_cl[i])
    return dict_cl


def non_iid_mnist(dataset, n_client):
    # 60000 images
    # 200 images per shard
    # 300 shards

    n_shards = 300
    n_imgs = 200
    idx_shard = [i for i in range(n_shards)]
    dict_cl = {i: np.array([]) for i in range(n_client)}
    idxs = np.arange(n_shards * n_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    print(idxs)

    # 2 shard per client

    for i in range(n_client):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_cl[i] = np.concatenate(
                (dict_cl[i], idxs[rand * n_imgs: (rand + 1) * n_imgs]), axis=0
            )
    return dict_cl


def non_iid_fmnist(dataset, n_client):
    # 60000 images
    # 200 images per shard
    # 300 shards

    n_shards = 300
    n_imgs = 200
    idx_shard = [i for i in range(n_shards)]
    dict_cl = {i: np.array([]) for i in range(n_client)}
    idxs = np.arange(n_shards * n_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    print("idxs",idxs)

    # 2 shard per client

    for i in range(n_client):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_cl[i] = np.concatenate(
                (dict_cl[i], idxs[rand * n_imgs: (rand + 1) * n_imgs]), axis=0
            )
        print("dict_cl[i]",dict_cl[i])
    return dict_cl


def non_iid_mnist_uneq(dataset, n_client):
    """
        Sample non-I.I.D client data from MNIST dataset s.t clients
        have unequal amount of data
        :param dataset:
        :param num_users:
        :returns a dict of clients with each clients assigned certain
        number of training imgs
        """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    # print("here")
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(n_client)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 2
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard + 1,
                                          size=n_client)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(n_client):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)
        print(random_shard_size)
        random_shard_size = random_shard_size - 1

        # Next, randomly assign the remaining shards
        for i in range(n_client):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)
    else:

        for i in range(n_client):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)
    # print(dict_users)
    return dict_users


# def non_iid_fmnist():

def non_iid_cifar(dataset, n_client):
    # 60000 images
    # 200 images per shard
    # 300 shards

    n_shards = 200
    n_imgs = 250
    idx_shard = [i for i in range(n_shards)]
    dict_cl = {i: np.array([]) for i in range(n_client)}
    idxs = np.arange(n_shards * n_imgs)
    labels = dataset.targets

    # sort labels

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    print(idxs)

    # 2 shard per client

    for i in range(n_client):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_cl[i] = np.concatenate(
                (dict_cl[i], idxs[rand * n_imgs: (rand + 1) * n_imgs]), axis=0
            )
    return dict_cl
