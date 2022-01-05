import pyro
import torch
import pickle

def model(data):
    ## Latent model parameters. Interested in infering rate
    with pyro.plate("starting_params"):
        ## Rate parameter
        rate0 = pyro.param("rate", torch.tensor(.5),
                           constraint=pyro.distributions.constraints.positive)

        r1 = pyro.sample("r1", pyro.distributions.Normal(.5, .01))
        rates = torch.tensor([r1, .1, .9])

        ## starting points
    s_start = pyro.sample("s_start", pyro.distributions.Normal(10, 0.01).expand([3]).to_event(1))
    # s_start = torch.tensor([10., 10., 10.])

    ## Loop over each observed trace
    with pyro.plate("sequences", data.shape[0]):
        ## Each step in trace
        for t in pyro.markov(range(1, data.shape[1])):
            # latent categorical sample
            sample = pyro.sample(f"sample_{t}",
                                 pyro.distributions.Categorical(s_start * rates),
                                 infer={"enumerate": "parallel"})

            ## Observed trace
            update = torch.zeros(3)
            update[sample] = 1
            s_temp = pyro.sample(f"s_{t}",
                                 pyro.distributions.Normal(s_start + update, .01).to_event(1),
                                 obs=data[:, t])

            s_start = s_temp


with open('data/test_data.pickle', 'rb') as handle:
    data = pickle.load(handle)

nuts_kernel = pyro.infer.NUTS(model)
mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=100)
mcmc.run(data)
samples = mcmc.get_samples()

results = mcmc.summary()
print(results)

with open('mcmc_results.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)