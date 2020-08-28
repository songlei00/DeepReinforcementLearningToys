from collections import namedtuple, deque
import random


class Memory:
    """Replay buffer for some off-policy algorithm

    Provide a simple way to store, sample and unpack the sampled data
    to a user-friendly form. 
    """

    def __init__(self, name, buffer_size, batch_size, *args):
        """Init the memory buffer

        Args:
            name: the name of sample data(namedtuple)
            buffer_size: max buffer size
            batch_size: the size of sample data
            args: indicate the data that you want to store, for example: 
            'state', 'action', 'reward', 'next_state'
        """
        self._memory = deque(maxlen=buffer_size)
        self._batch_size = batch_size
        self._experience = namedtuple(name, args)

    def push(self, *args):
        self._memory.append(self._experience(*args))

    def sample(self, n=None, is_unpacked=True):
        """Sample batch size from memory

        Args:
            n: the size of sample batch, default to batch_size
            is_unpacked: control whether the return will be a 
            full trace data or divide into state, action, reward, ...

        Returns:
            if the parameter is_unpacked is True, return a tuple of 
            (state, action, reward, next_state, ...)

            if the parameter is_unpacked is False, return a tuple of 
            namedtuple, each namedtuple is a sample obtained by interacting 
            with the environment.

        Well, I think it is unreadable...
        """
        sample_size = n if n != None else self._batch_size
        data = random.sample(self._memory, k=sample_size)
        if is_unpacked:
            # zip(*data) will 'transpose' the memory, unpack and pass the 
            # transposed data to self._experience. We can obtain all state using
            # data.state, all action using data.action and so on.
            data = self._experience(*zip(*data))
        return data

    def __len__(self):
        return len(self._memory)

