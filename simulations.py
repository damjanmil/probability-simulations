import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson, norm, expon


def simulate_binomial(n=10, p=0.5, size=10_000):
    samples = np.random.binomial(n=n, p=p, size=size)

    x = np.arange(0, n + 1)
    theoretical = binom.pmf(x, n, p)

    plt.figure(figsize=(8, 5))
    plt.hist(samples, bins=np.arange(-0.5, n + 1.5, 1), density=True, alpha=0.6, label="Simulation")
    plt.plot(x, theoretical, "ro-", label="Theoretical PMF")
    plt.title(f"Binomial Distribution: n={n}, p={p}")
    plt.xlabel("Number of successes")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("plots/binomial_distribution.png")
    plt.show()


def simulate_poisson(lam=4, size=10_000):
    samples = np.random.poisson(lam=lam, size=size)

    x = np.arange(0, max(samples) + 1)
    theoretical = poisson.pmf(x, lam)

    plt.figure(figsize=(8, 5))
    plt.hist(samples, bins=np.arange(-0.5, max(samples) + 1.5, 1), density=True, alpha=0.6, label="Simulation")
    plt.plot(x, theoretical, "ro-", label="Theoretical PMF")
    plt.title(f"Poisson Distribution: lambda={lam}")
    plt.xlabel("Number of events")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("plots/poisson_distribution.png")
    plt.show()


def simulate_normal(mu=0, sigma=1, size=10_000):
    samples = np.random.normal(loc=mu, scale=sigma, size=size)

    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
    theoretical = norm.pdf(x, mu, sigma)

    plt.figure(figsize=(8, 5))
    plt.hist(samples, bins=50, density=True, alpha=0.6, label="Simulation")
    plt.plot(x, theoretical, "r-", label="Theoretical PDF")
    plt.title(f"Normal Distribution: mu={mu}, sigma={sigma}")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("plots/normal_distribution.png")
    plt.show()


def simulate_exponential(lam=1.5, size=10_000):
    scale = 1 / lam
    samples = np.random.exponential(scale=scale, size=size)

    x = np.linspace(0, max(samples), 300)
    theoretical = expon.pdf(x, scale=scale)

    plt.figure(figsize=(8, 5))
    plt.hist(samples, bins=50, density=True, alpha=0.6, label="Simulation")
    plt.plot(x, theoretical, "r-", label="Theoretical PDF")
    plt.title(f"Exponential Distribution: lambda={lam}")
    plt.xlabel("Waiting time")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("plots/exponential_distribution.png")
    plt.show()


if __name__ == "__main__":
    simulate_binomial()
    simulate_poisson()
    simulate_normal()
    simulate_exponential()