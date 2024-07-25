import requests
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.graph import Apply
import datetime 
import time
import os

model_url = os.environ.get("MODEL_URL", "http://localhost:4242")

def patch_requests_post_default_timeout(timeout=5.0):
    original_post = requests.post

    def post_with_timeout(*args, **kwargs):
        # Explicitly set timeout to post /Evaluate
        if 'Evaluate' in args[0] and 'timeout' not in kwargs:
            kwargs['timeout'] = timeout
        return original_post(*args, **kwargs)

    requests.post = post_with_timeout

patch_requests_post_default_timeout(timeout=40.0)

import umbridge

print(f"Running on PyMC v{pm.__version__}")

umbridge_model = umbridge.HTTPModel(model_url, "forward")
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

real_rho = 2.6
real_cp = 4.0
real_cs = 2.0
real_sigma = 1

real_output = np.array(umbridge_model([[real_rho, real_cp, real_cs]]))

def apply_umbridge_model(params):
    begin = datetime.datetime.now()
    try:
        start = time.perf_counter()
        begin = datetime.datetime.now()
        model = np.array(umbridge_model([params]))
        end = time.perf_counter()
        timespent = end - start
        if timespent > 10:
            print(f"begin: {begin}, apply_model takes       >>> {timespent:0.4f} <<<      seconds, with params:  ", params)
        else:
            print(f"spent    >>> {timespent:0.4f} <<<   seconds")
        return model
    except:
        print(f"apply_umbridge_model failed, begin: {begin}, params: {params} retrying ...................................")
        time.sleep(61)
        return apply_umbridge_model(params)

def loh_model(rho, cp, cs):
    return apply_umbridge_model([rho.item(), cp.item(), cs.item()])

def loh_loglike(rho, cp, cs, sigma, data):
    try:
        model = loh_model(rho, cp, cs)
        has_none = np.equal(model, None).any()
        if has_none:
            print('has none !!!!!!!!!!')
            return np.array([np.float64(-100)])
        residuals = data[:, 1:] - model[:, 1:]
        residuals = residuals.flatten()
        residual2 = np.square(residuals)
        residualSq = np.sqrt(residual2)
        log_like = -np.sum(residualSq)

        has_infinity = np.isinf(log_like)
        # For array log_like
        # has_infinity = np.isinf(log_like).any()

        if has_infinity:
            log_like = np.float64(-100)
            # For array log_like
            # log_like = np.full(n, np.float64(-100))
            print('log_like is inf !!!!!!!!!!!!!!!!!!!')
            print('rho', rho)
            print('cp', cp)
            print('cs', cs)
        print('rho | cp | cs | sigma       ', rho, '|', cp, '|', cs, '|', sigma, '     log_like: ', log_like, '   at: ', datetime.datetime.now())
        # print('rho | cp | cs | sigma       ', rho, '|', cp, '|', cs, '|', sigma, '   at: ', datetime.datetime.now())
        return np.array([log_like])
    except:
        print("Caught exception !!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print('rho | cp | cs | sigma       ', rho, '|', cp, '|', cs, '|', sigma, '   at: ', datetime.datetime.now())
        log_like = np.float64(-100)
        return np.array([log_like])
    

class LogLike(pt.Op):
    def make_node(self, rho, cp, cs, sigma, data) -> Apply:
        rho = pt.as_tensor(rho)
        cp = pt.as_tensor(cp)
        cs = pt.as_tensor(cs)
        sigma = pt.as_tensor(sigma)
        data = pt.as_tensor(data)

        inputs = [rho, cp, cs, sigma, data]
        outputs = [pt.vector()]
        return Apply(self, inputs, outputs)

    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
        rho, cp, cs, sigma, data = inputs
        loglike_eval = loh_loglike(rho, cp, cs, sigma, data)
        outputs[0][0] = np.asarray(loglike_eval)

loglike_op = LogLike()



def custom_dist_loglike(data, rho, cp, cs, sigma):
    return loglike_op(rho, cp, cs, sigma, data)

if __name__ == "__main__":
    with pm.Model() as exaseis_model:
        # sigma = pm.HalfNormal("sigma", sigma=1)
        sigma = np.array([1])
        # rho =  pm.Gamma("rho", mu=1, sigma=10)
        # cp =  pm.Gamma("cp", mu=1, sigma=10)
        # cs =  pm.Gamma("cs", mu=1, sigma=10)

        interval = pm.distributions.transforms.Interval(lower=0.01, upper=25)
        rho =  pm.Normal("rho", mu=1, sigma=5, transform=interval)
        cp =  pm.Normal("cp", mu=1, sigma=5, transform=interval)
        cs =  pm.Normal("cs", mu=1, sigma=5, transform=interval)
        # sigma = pm.Normal("sigma", mu=1, sigma=5, transform=interval)
        likelihood = pm.CustomDist('likelihood', rho, cp, cs, sigma, observed=np.array(real_output), logp=custom_dist_loglike)
    with exaseis_model:
        # step = pm.Slice()
        step = pm.Metropolis()
        idata = pm.sample(300, tune=150, step=step)


    print("Finished!!!")

    az.plot_trace(idata, combined=True)
    # az.summary(idata, round_to=2)
    idata.to_netcdf("result.nc")
    plt.savefig('result.png')