from collections import namedtuple, deque
import random


class Memory:
    def __init__(self, maxlen, sample_size):
        self.experience = namedtuple(
            'experience',
            (
                'state',
                'action',
                'reward',
                'next_state',
                'done',
            )
        )
        self.memory = deque(maxlen=maxlen)
        self.sample_size = sample_size

    def push(self, *args):
        self.memory.append(self.experience(*args))

    def sample(self):
        return self.experience(*zip(*random.sample(self.memory, self.sample_size)))

    def __len__(self):
        return len(self.memory)


if __name__ == '__main__':
    memory = Memory(3, 2)
    memory.push('s1', 1, 2, 's2')
    memory.push('s2', 3, 4, 's3')
    memory.push('s3', 5, 6, 's4')
    data = memory.sample()
    print(data.state)
