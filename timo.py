
import numpy as np
import matplotlib.pyplot as plt
import pymc
import json
import seaborn as sns
import arviz as az
import time

from pytensor.compile.ops import as_op
import pytensor.tensor as tt
import pytensor

# for debugging:
# pytensor.config.optimizer='None'

basic_model = pymc.Model()

import subprocess
import numpy as np
import matplotlib.pyplot as plt
import json

ranks = 24
release_mode = True
build_dir = "/home/asaxena/aspect/build/"


def send(process, message):
    print(">", message)
    process.stdin.write((message+"\n").encode())
    process.stdin.flush()
    
def receive(process):
    line = process.stdout.readline().decode()
    line = line[:-1] # strip endline
    print("|", line)

    return line

def get_residual(process):
    value = 0.
    while True:
        line = receive(process)
        if line.startswith("     Velocity"):
            word = line.split()
            value = float(word[-1])
            return value

def wait_for_prompt(process):
    while True:
        line = receive(process)
        if line=="?":
            return

minerr = 1e100

def modify_world_builder_file(input_wb_file, new_wb_parameter_value):
    with open (input_wb_file) as f:
        wb_template = json.load(f)
        wb_models   = wb_template['features']
        for model in wb_models:
            if (model['name'] == "Subducting plate"):
                for segment in model["segments"]:
                    segment["thickness"] = [new_wb_parameter_value]

        all_faults_json = {
        "version": "0.5", 
        "cross section":[[0,0],[100,0]],
        "features": wb_models}

    with open ("temp-2.wb", "w") as f2:
        json.dump(all_faults_json, f2)  

binary = build_dir + "aspect-release" if release_mode else "aspect"
process =  subprocess.Popen(["/usr/bin/mpirun", "-np",  str(ranks), "/home/asaxena/aspect/build/aspect-release", "4.prm"], 
                        stdin = subprocess.PIPE,
                        stdout = subprocess.PIPE)

thickness_stats = []
alpha_stats     = []
residual_stats  = []

def get_error(thickness, alpha):
    modify_world_builder_file('simple.wb', float(thickness)*1e3)
    thickness_stats.append(float(thickness)*1e3)
        
    send(process, "wb temp-2.wb")
    wait_for_prompt(process)

    send(process, "thermal-exp %s" %(float(alpha)))
    alpha_stats.append(float(alpha))
    wait_for_prompt(process)

    send(process, "continue")
    residual = get_residual(process)*1e3 #mm/year
    print ('residual', residual)
    residual_stats.append(residual)

    return residual

@as_op(itypes=[tt.dscalar, tt.dscalar], otypes=[tt.dscalar])
def theta(test_thickness_values, test_alpha_values):
    global minerr
    x, y = test_thickness_values, test_alpha_values
    try:
        e = get_error(x, y)
        sigma = 1e-2
        if e < minerr:
            minerr = e
            print("best fit", e, "values:", x, y)
        return np.array(1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*(e*e/sigma*sigma)))
    except ValueError:
        return np.array(-1e20)  # float("inf")

start_time = time.time()

with basic_model:
    # Priors for unknown model parameters
    test_thickness_values = pymc.Uniform("thickness_value", lower=60., upper=220.)
    test_alpha_values     = pymc.Uniform("alpha_value", lower=1e-6, upper=1e-4)

    # this is the sampling distribution of the output (residual/misfit)
    misfit = pymc.Potential("likelihood", theta(test_thickness_values, test_alpha_values))

    step = pymc.Metropolis()
    trace = pymc.sample(draws=2000, tune=1, step=step, cores=1, chains=1, return_inferencedata=True)

    print("--- %s seconds ---" % (time.time() - start_time))

    print(az.summary(trace, kind="stats"))
    az.plot_trace(trace, var_names=["thickness_value", "alpha_value"])
    plt.savefig("2D_prm_plot.pdf", dpi=150)
    trace_dict = trace.to_dict()

    #plt.figure(2)
    az.plot_posterior(trace)
    plt.savefig("plot_posterior.pdf", dpi=150)
    az.plot_pair(trace,kind='kde')
    plt.savefig("plot_trace_pair.pdf", dpi=150)

    #plt.show
plt.rcParams.update({'font.size': 15})   
data_prior_sampled = np.column_stack((thickness_stats, alpha_stats, residual_stats))
data_posterior_sampled = []

for i in range(len(trace.posterior.thickness_value[0][:].values)):
    # there are multiple values for the same thickness, so we need to find index common to both angle
    # and thickness, use just one of the values
    indx = np.where( (thickness_stats == trace.posterior.thickness_value[0][i].values*1e3) & 
                    (alpha_stats == trace.posterior.alpha_value[0][i].values) )[0][0]
    data_posterior_sampled.append(data_prior_sampled[indx, :][:].tolist())

data_posterior_arr = np.array(data_posterior_sampled)

fig, (ax3, ax4) = plt.subplots(nrows=1, ncols=2, figsize=(16,6))
plot_stats = ax3.scatter(np.array(thickness_stats)/1e3, alpha_stats, c=np.log(residual_stats))
ax3.scatter(100, 4e-5, c='r', marker='x')
ax3.title.set_text('Prior')
ax3.set_xlabel('Fault thickness [km]')
ax4.set_ylabel('Thermal exapansivity [1/K]')
cbar = fig.colorbar(plot_stats, ax=ax3, label='log(Residual) [mm/year]')

plot_stats1 = ax4.scatter(data_posterior_arr[:,0]/1e3, data_posterior_arr[:,1], c=np.log(data_posterior_arr[:,2]))
ax4.scatter(100, 4e-5, c='r', marker='x')
ax4.set_xlim(40, 220)
ax4.set_ylim(5e-6, 5e-5)
ax4.title.set_text('Sampled')
ax4.set_xlabel('Fault thickness [km]')
ax4.set_ylabel('Thermal exapansivity [1/K]')
cbar = fig.colorbar(plot_stats1, ax=ax4, label='log(Residual) [mm/year]')
plt.savefig("residual_posterior.pdf", dpi=150)
