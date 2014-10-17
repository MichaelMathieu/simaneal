import os
import datetime
import threading
import math
import string

torchpath = '/home/myrhev/libs/torch/torch/installed/bin/torch-qlua'

ncores = 8
nseeds = 1000
nsamples = 6000
niter = 70000
teta = 0.5
neta = 0.01
tgamma = 0.75
nhid = 5

def f(istart, iend):
    for i in range(istart, iend):
        seed = i
        jobname = 'mnist_nhid-%d_nsamples-%d_niter-%d_teta-%f_neta-%f_tgamma-%f_seed-%d__%s'%(nhid, nsamples, niter, teta, neta, tgamma, seed, string.replace(str(datetime.datetime.now()), ' ', '-'))
        outputpath = 'outputs/' + jobname + '.txt'
        print(jobname)
        os.system('%s mnist.lua -nsamples %d -niter %d -teta -%f -neta %f -tgamma %f -nhid %d -seed %d -jobname %s > %s'%(torchpath, nsamples, niter, teta, neta, tgamma, nhid, seed, jobname, outputpath))

iincr = int(math.ceil(nseeds/ncores))
for i in range(ncores):
    istart = i*iincr
    iend = min(nseeds, istart + iincr)
    t = threading.Thread(target=f, args=(istart, iend))
    t.start()

