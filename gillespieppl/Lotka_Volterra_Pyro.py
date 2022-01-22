import pyro
import torch
import pickle

## Define Model
def lotka_volterra_0(data):
    collect = list()

    num_sequences, max_length, data_dim = data.shape

    ## Define true rates
    rates = torch.tensor([.7, .3, .9])

    ## Loop over sequences
    sample_plate = pyro.plate("sample", data_dim, dim=-1)
    for i in pyro.plate("sequences", num_sequences):
        temp_list = list()
        s_start = torch.tensor([[10., 10.]])
        for t in pyro.markov(torch.arange(max_length)):
            rates_matrix = torch.stack([rates[0] * pyro.ops.indexing.Vindex(s_start)[..., :, 0],
                                        rates[1] * pyro.ops.indexing.Vindex(s_start)[..., :, 0
                                                   ] * pyro.ops.indexing.Vindex(s_start)[..., :, 1],
                                        rates[2] * pyro.ops.indexing.Vindex(s_start)[..., :, 1]], axis=1)

            x = pyro.sample(
                "x_{}_{}".format(t, i),
                pyro.distributions.Categorical(rates_matrix),
                infer={"enumerate": "parallel"},
            )
            ## Update tensor
            transition = torch.tensor([[1, 0],
                                       [-1, 1],
                                       [0, -1]])
            update = transition[x]
            with sample_plate:
                # print(pyro.sample(
                #     "me_{}_{}".format(t, i),
                #     pyro.distributions.Normal(s_start + update, .01)).squeeze().shape)
                # print(data[i, t].shape)
                ## Observed sample
                obs = pyro.sample(
                    "y_{}_{}".format(t, i),
                    pyro.distributions.Normal(s_start + update, .01),
                    obs=data[i, t]
                ).squeeze()
                # print(obs)
                temp_list.append(obs.squeeze())

            ## True update
            s_start = pyro.deterministic("d_{}_{}".format(t, i), s_start + update, event_dim=1)
        # print(temp_list)
        collect.append(torch.stack(temp_list))
    return collect


if __name__ == '__main__':

    with open(r'../data/lv_data.pickle', "rb") as input_file:
        temp = pickle.load(input_file)

    hmm_guide = pyro.infer.autoguide.AutoDelta(pyro.poutine.block(lotka_volterra_0, expose=["rates"]))
    pyro.clear_param_store()
    elbo = pyro.infer.TraceEnum_ELBO(max_plate_nesting=3)
    print(elbo.loss(lotka_volterra_0, hmm_guide, temp)) ## Ensure model compiles