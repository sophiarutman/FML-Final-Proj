import numpy as np

class TabularQLearner:
    def __init__ (self, states = 100, actions = 4, alpha = 0.2, gamma = 0.9, epsilon = 0.98, epsilon_decay = 0.999, dyna = 0):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.dyna = dyna
        self.state = 0
        self.action = 0
        self.QTable = np.random.random((states, actions))
        self.experiences = []
        

    def train (self, s, r):
        retained = (1 - self.alpha) * self.QTable[self.state][self.action]
        epsCheck = np.random.rand()
        if epsCheck < self.epsilon:
            a = np.random.randint(0, self.actions)
            self.epsilon *= self.epsilon_decay
        else:
            a = np.argmax(self.QTable[s])
        update = self.alpha * (r + self.gamma * self.QTable[s, a])
        self.QTable[self.state][self.action] = retained + update
        self.experiences.append((self.state, self.action, s, r))
        self.state = s
        self.action = a
        i = 0
        while i < self.dyna:
            sample = self.experiences[np.random.randint(0, len(self.experiences))]
            dynaS, dynaA, dynaSPrime, dynaR = sample
            retained = (1 - self.alpha) * self.QTable[dynaS][dynaA]
            if epsCheck < self.epsilon:
                dynaAPrime = np.random.randint(0, self.actions)
                self.epsilon *= self.epsilon_decay
            else:
                dynaAPrime = np.argmax(self.QTable[dynaSPrime])
            update = self.alpha * (dynaR + self.gamma * self.QTable[dynaSPrime, dynaAPrime])
            self.QTable[dynaS][dynaA] = retained + update
            i += 1
        return a

    def test (self, s):
        a = np.argmax(self.QTable[s])
        self.state = s
        self.action = a
        return a