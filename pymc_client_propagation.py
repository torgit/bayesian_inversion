import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import umbridge
import concurrent.futures
import os

model_url = os.environ.get("MODEL_URL", "http://localhost:4242")

print(f"Running on PyMC v{pm.__version__}")

umbridge_model = umbridge.HTTPModel(model_url, "forward")
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

real_v0 = 4.139874e-20
real_v1 = -0.001686105
real_v2 = -1.662302e-20

output_size = 123
sampling_size = 50
real_rho = 2.6
real_cp = 4.0
real_cs = 2.0
real_sigma = 1
cp_s = real_cp
cp_s = np.round(cp_s, decimals=2)
cs_s = real_cs + rng.normal(size=sampling_size) * real_sigma
cs_s = np.round(cs_s, decimals=2)

v0_s = np.empty(sampling_size)
v1_s = np.empty(sampling_size)
v2_s = np.empty(sampling_size)

for i in range(sampling_size):
    v0_s[i] = real_v0
    v1_s[i] = real_v1
    v2_s[i] = real_v2

with pm.Model() as exaseis_model:
    sigma = pm.HalfNormal("sigma", sigma=1)
    rho =  pm.Gamma("rho", mu=real_rho, sigma=1)
    cp =  pm.Gamma("cp", mu=real_cp, sigma=0.00001)
    cs =  pm.Gamma("cs", mu=real_cs, sigma=0.00001)

    # Expected value of outcome
    v0 = np.empty(sampling_size)
    v1 = np.empty(sampling_size)
    v2 = np.empty(sampling_size)

rho_s, cp_s, cs_s = pm.draw([rho, cp, cs], draws=sampling_size)

def parallel_function(x):
    return umbridge_model([[x, real_cp, real_cs]])

results = []

with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
    results = executor.map(parallel_function, rho_s)

time = np.zeros(output_size)
v0_s = np.zeros((output_size, sampling_size))
v1_s = np.zeros((output_size, sampling_size))
v2_s = np.zeros((output_size, sampling_size))

# Sampling loop
for i, result in enumerate(results):
    # Output loop (timespan)
    for j, result_j in enumerate(result):
        time[j] = result_j[0]
        v0_s[j][i] = result_j[1]
        v1_s[j][i] = result_j[2]
        v2_s[j][i] = result_j[3]

v0_s_mean = np.mean(v0_s, axis=1)
v0_s_std = np.std(v0_s, axis=1)

v1_s_mean = np.mean(v1_s, axis=1)
v1_s_std = np.std(v1_s, axis=1)

v2_s_mean = np.mean(v2_s, axis=1)
v2_s_std = np.std(v2_s, axis=1)

print(v0_s_mean)

fig, ax = plt.subplots()
ax.plot(time, v0_s_mean, 'k-')
ax.fill_between(time, v0_s_mean - v0_s_std, v0_s_mean + v0_s_std, alpha=0.3)


fig, ax2 = plt.subplots()
ax2.plot(time, v1_s_mean, 'k-')
ax2.fill_between(time, v1_s_mean - v1_s_std, v1_s_mean + v1_s_std, alpha=0.3)


fig, ax3 = plt.subplots()
ax3.plot(time, v2_s_mean, 'k-')
ax3.fill_between(time, v2_s_mean - v2_s_std, v2_s_mean + v2_s_std, alpha=0.3)

print("v0 mean - ", np.mean(v0_s_mean))
print("v1 mean - ", np.mean(v1_s_mean))
print("v2 mean - ", np.mean(v2_s_mean))
plt.show()