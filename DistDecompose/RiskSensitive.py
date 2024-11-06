import cmath
import numpy as np
import torch as T
from torch.distributions import Normal, Beta, Uniform, Gamma, Exponential, Chi2
import matplotlib.pyplot as plt
import scipy.stats as stats

class RiskSensitive:
    def __init__(self, remapping, eta, discrete_num = 256, quantile = 3) -> None:
        self.remapping = remapping
        self.eta = eta
        self.discrete_num = discrete_num
        self.quantile = quantile
        pass

    def calExpectation(self, mu, sigma):
        x = np.linspace(mu - self.quantile * sigma, mu + self.quantile * sigma, self.discrete_num)
        y = sum(self.reProbForGuass(x, mu, sigma) * x * 2 * self.quantile * sigma / self.discrete_num)
        # print(y)
        
        # plt.plot(x, self.reProbForGuass(x, mu, sigma) * x * 2 * self.quantile / self.discrete_num)
        # plt.show()
        
        return y

    def reProbForGuass(self, x, mu, sigma):
        prob = stats.norm.pdf(x, mu, sigma)
        c_prob = stats.norm.cdf(x, mu, sigma)
        re_prob = prob
        
        if self.remapping == 'Normal':
            re_prob = prob
        elif self.remapping == 'CPW':
            re_prob = self.div_CPW(c_prob) * prob
        elif self.remapping == 'POW':
            re_prob = self.div_POW(c_prob) * prob
        elif self.remapping == 'WANG':
            re_prob = self.div_WANG(c_prob) * prob
        elif self.remapping == 'CVaR':
            re_prob = self.div_CVaR(c_prob) * prob        
        return T.tensor(re_prob)

    def CPW(self, tau):
        return tau ** self.eta / (tau ** self.eta + (1 - tau) ** self.eta) ** (1 / self.eta)

    def div_CPW(self, tau):
        tau = np.clip(tau, 0.0001, 0.9999)
        _tmp = tau ** self.eta + (1 - tau) ** self.eta
        return (self.eta * tau ** (self.eta - 1) - 1 / self.eta / _tmp * tau ** self.eta * (self.eta * tau ** (self.eta - 1) - self.eta * (1 - tau) ** (self.eta - 1))) / _tmp ** (1 / self.eta)

    def WANG(self, tau):
        return stats.norm.cdf(stats.norm.ppf(tau) + self.eta, 0, 1)

    def div_WANG(self, tau):
        return stats.norm.pdf(stats.norm.ppf(tau, 0, 1) + self.eta, 0, 1) / stats.norm.pdf(stats.norm.ppf(tau, 0, 1), 0, 1)

    def POW(self, tau):
        if self.eta >= 0:
            return tau ** (1 / (1 + abs(self.eta)))
        else:
            return 1 - (1 - tau) ** (1 / (1 + abs(self.eta)))

    def div_POW(self, tau):
        if self.eta >= 0:
            return (1 / (1 + abs(self.eta))) * tau ** (-abs(self.eta) / (1 + abs(self.eta)))
        else:
            return (1 / (1 + abs(self.eta))) * (1 - tau) ** (-abs(self.eta) / (1 + abs(self.eta)))

    def CVaR(self, tau):
        return self.eta * tau

    def div_CVaR(self, tau):
        return self.eta
    
# aaa = RiskSensitive("Normal", 0.74)
# aaa.calExpectation(20, 10)