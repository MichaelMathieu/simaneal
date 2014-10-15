require 'simaneal'
require 'nn'
require 'paths'
require 'image'
require 'io'

cmd = torch.CmdLine()
cmd:option('-seed', 1, 'Manual seed')
cmd:option('-nhid', 16, 'Number of hidden units')
cmd:option('-teta', 30, "Temperature eta")
cmd:option('-tgamma', 1, "Temperature gamma")
cmd:option('-neta', 20, "Neighbour eta")
cmd:option('-nchanges', 20, "Neighbour number of changes")
cmd:option('-niter', 100000, "Number of iterations")
cmd:option('-nsamples', 6000, "Number of samples")
cmd:option('-wndisc', 100, "Number of different weights")
cmd:option('-wrange', 10, "Range of weights")
cmd:option('-jobname', 'job', 'Job name')
opt = cmd:parse(arg)

-- parameters
local params = {
   seed = tonumber(opt.seed),
   nHidden = tonumber(opt.nhid),
   jobname = opt.jobname,
   nSamples = tonumber(opt.nsamples),
   nIter = tonumber(opt.niter),
   temp_eta = tonumber(opt.teta),
   temp_gamma = tonumber(opt.tgamma),
   neighbour_eta = tonumber(opt.neta),
   neighbour_n_changes = tonumber(opt.nchanges),
   weight_n_disc = tonumber(opt.wndisc),
   weight_range = tonumber(opt.wrange)
}

torch.manualSeed(params.seed)

local nInputX = 10
local nInputY = 10
local nInput = nInputX*nInputY
local nHidden = params.nHidden
local nOutput = 10
local model = nn.Sequential()
model:add(nn.Linear(nInput, nHidden))
model:add(nn.Threshold())
model:add(nn.Linear(nHidden, nOutput))
--model:add(nn.Linear(nInput, nOutput))
model:add(nn.LogSoftMax())
local criterion = nn.ClassNLLCriterion()

local savename = 'small_mnist_'..(nInputX*nInputY)..'.t7b'
local dataset
if not paths.filep(savename) then
   local full_dataset = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/mnist-torch7/train_28x28.th7nn')
   dataset = {}
   dataset.data = torch.Tensor(full_dataset.data:size(1), nInput)
   dataset.labels = full_dataset.labels
   for i = 1, full_dataset.data:size(1) do
      local rescaled = image.scale(full_dataset.data[i], nInputX, nInputY, 'bilinear')
      dataset.data[i]:copy(rescaled:reshape(nInput))
   end
   torch.save(savename, dataset, 'binary')
else
   dataset = torch.load(savename)
end
local savename = 'small_mnist_test_'..(nInputX*nInputY)..'.t7b'
local dataset_test
if not paths.filep(savename) then
   local full_dataset = torch.load('/misc/vlgscratch3/LecunGroup/mbhenaff/mnist-torch7/test_28x28.th7nn')
   dataset_test = {}
   dataset_test.data = torch.Tensor(full_dataset.data:size(1), nInput)
   dataset_test.labels = full_dataset.labels
   for i = 1, full_dataset.data:size(1) do
      local rescaled = image.scale(full_dataset.data[i], nInputX, nInputY, 'bilinear')
      dataset_test.data[i]:copy(rescaled:reshape(nInput))
   end
   torch.save(savename, dataset_test, 'binary')
else
   dataset_test = torch.load(savename)
end

dataset.labels = dataset.labels:double()

local n_total = params.nSamples
dataset.data   = dataset.data  [{{1, n_total}, {}}]
dataset.labels = dataset.labels[{{1, n_total}}]

local legal_weights = torch.Tensor(params.weight_n_disc)
local w_step = params.weight_range * 2 / params.weight_n_disc
for i = 1, params.weight_n_disc do
   legal_weights[i] = - params.weight_range + w_step * i
end

local nSamples = dataset.labels:size(1)
local batch_size = nSamples
local w, dw = model:getParameters()
for i = 1, w:size(1) do
   w[i] = legal_weights[torch.random(legal_weights:size(1))]
end
local w_min = legal_weights[1]
local w_max = legal_weights[legal_weights:size(1)]
local initial_state = w

local function getNeighbour(p, i_step)
   local a
   if i_step > params.nIter / 3 then
      a = params.neighbour_eta/10
   else
      a = params.neighbour_eta
   end
   local p2 = p:clone()
   for i = 1, params.neighbour_n_changes do
      local delta = math.round(torch.normal():mul(a)) / w_step
      local k = torch.random(p2:size(1))
      p2[k] = math.max(w_min, math.min(w_max, p2[k] + delta))
   end
   return p2
end
local perm = torch.randperm(nSamples)
local input = torch.Tensor(batch_size, nInput)
local target = torch.Tensor(batch_size)
local old_w = w:clone()

local function energy(p, i_step, this_batch_size)
   old_w:copy(w)
   w:copy(p)
   print("ok")
   local output = model:forward(dataset.data)
   print("done")
   local err = criterion:forward(output, dataset.labels)
   w:copy(old_w)
   return err
end

local function temperature(i_step)
   return params.temp_eta/(1+math.pow(i_step, params.temp_gamma))
end

io.write("Initial energy : " .. energy(initial_state, 1, nSamples) .. "\n")
local best_state, best_energy = simaneal(initial_state, getNeighbour,
					 energy, temperature, params.nIter)
io.write("Final energy : " .. energy(best_state, 1, nSamples) .. "\n")
io.flush()

w:copy(best_state)
local output = model:forward(dataset.data)
err = criterion:forward(output, dataset.labels)
io.write("Final training err : " .. err .. "\n")


-- sanity check
local w, dw = model:getParameters()
criterion.sizeAverage = false
local learning_rate = 0.01
local batch_size = 16
local target = torch.Tensor(batch_size)
local input = torch.Tensor(batch_size, nInput)
local nSamples = dataset.labels:size(1)

--[[
for i = 1, 15 do
   local perm = torch.randperm(nSamples)
   local avg_err = 0
   for j = 1, (nSamples-batch_size), batch_size do
      for k = 1, batch_size do
	 input[k]:copy(dataset.data[perm[j+k-1] ])
	 target[k] = dataset.labels[perm[j+k-1] ]
      end
      dw:zero()
      local output = model:forward(input)
      local err = criterion:forward(output, target)
      local derr_do = criterion:backward(output, target)
      model:backward(input, derr_do)
      w:add(-learning_rate, dw)
      avg_err = avg_err + err
   end
   avg_err = avg_err / nSamples
   print("train err", avg_err)

   avg_err = 0
   local n_good = 0
   for j = 1, (dataset_test.data:size(1)-batch_size), batch_size do
      for k = 1, batch_size do
	 input[k]:copy(dataset_test.data[j+k-1])
	 target[k] = dataset_test.labels[j+k-1]
      end
      local output = model:forward(input)
      local err = criterion:forward(output, target)
      local _, m = output:max(2)
      n_good = n_good + m:squeeze():eq(target:long()):sum()
      avg_err = avg_err + err
   end
   avg_err = avg_err / dataset_test.data:size(1)
   print("test err", avg_err)
   print("test acc (%)", 100 * n_good / dataset_test.data:size(1))
end
--]]

avg_err = 0
local n_good = 0
for j = 1, (dataset_test.data:size(1)-batch_size), batch_size do
   for k = 1, batch_size do
      input[k]:copy(dataset_test.data[j+k-1])
      target[k] = dataset_test.labels[j+k-1]
   end
   local output = model:forward(input)
   local err = criterion:forward(output, target)
   local _, m = output:max(2)
   n_good = n_good + m:squeeze():eq(target:long()):sum()
   avg_err = avg_err + err
end
avg_err = avg_err / dataset_test.data:size(1)
io.write("Final testing error : " .. avg_err .. "\n")
local accuracy = n_good / dataset_test.data:size(1)
io.write("final testing accuracy : " .. (100 * accuracy) .. "\n")
io.flush()

io.write("\n\n\n")
for k = 1, best_state:size(1) do
   io.write("" .. best_state[k] .. " ")
end
io.write("\n")
io.flush()

torch.save('/home/mfm352/phd/annealing/outputs/test_'..params.jobname..'.t7b',
	   {state=best_state,
	    training_error=best_energy,
	    testing_error=avg_err,
	    testing_acc=accuracy})
