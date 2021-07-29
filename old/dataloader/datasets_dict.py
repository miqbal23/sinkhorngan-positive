datasets_dict = {
    "mnist": {
        "type": "mnist",
        "num_workers": 2,
        "input_size": (32, 32),
        "n_channel": 1,
        "shuffle": True,
    },
    "cifar10": {
        "type": "cifar10",
        "num_workers": 2,
        "input_size": (32, 32),
        "n_channel": 3,
        "shuffle": True,
    },
    "cifar100": {
        "type": "cifar100",
        "num_workers": 2,
        "input_size": (32, 32),
        "n_channel": 3,
        "shuffle": True,
    },
    "dogvscat": {
        "type": "dogvscat",
        "num_workers": 2,
        "input_size": (32, 32),
        "n_channel": 3,
        "shuffle": True,
    },
    "cat": {
        "type": "cat",
        "num_workers": 2,
        "input_size": (32, 32),
        "n_channel": 3,
        "shuffle": True,
    },
    "celeba": {
        "type": "celeba",
        "num_workers": 2,
        "input_size": (32, 32),
        "n_channel": 3,
        "shuffle": True,
        "crop_size": 160,
    }
}
