require 'simaneal'
require 'nn'
require 'paths'
require 'image'

torch.manualSeed(1)

local nInputX = 10
local nInputY = 10
local nInput = nInputX*nInputY
local nHidden = 16
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

local n_total = 6000
dataset.data   = dataset.data  [{{1, n_total}, {}}]
dataset.labels = dataset.labels[{{1, n_total}}]

local nSamples = dataset.labels:size(1)
local batch_size = nSamples
local w, dw = model:getParameters()
local initial_state = w
local rnd = torch.Tensor(w:size())
local function getNeighbour(p)
   rnd:normal():mul(0.01)--math.sqrt(0.1/nSamples))
   return p + rnd
end
local perm = torch.randperm(nSamples)
local input = torch.Tensor(batch_size, nInput)
local target = torch.Tensor(batch_size)
local old_w = w:clone()
--[[local function energy(p, i_step, this_batch_size)
   this_batch_size = this_batch_size or batch_size
   old_w:copy(w)
   w:copy(p)

   input:resize(this_batch_size, nInput)
   target:resize(this_batch_size)

   --if torch.uniform() < batch_size / nSamples then
     -- perm = torch.randperm(nSamples)
      --print("regen", batch_size / nSamples)
   --end
   
   for k = 1, this_batch_size do
      input[k]:copy(dataset.data[perm[1 + k % nSamples] ])
      target[k] = dataset.labels[perm[1 + k % nSamples] ]
   end

   local output = model:forward(input)
   local err = criterion:forward(output, target)
   w:copy(old_w)
   return err
   end--]]
local function energy(p, i_step, this_batch_size)
   this_batch_size = this_batch_size or batch_size
   old_w:copy(w)
   w:copy(p)
   local output = model:forward(dataset.data)
   local err = criterion:forward(output, dataset.labels)
   w:copy(old_w)
   return err
end
local function temperature(i_step)
   return 30/(1+i_step)
end

print(energy(initial_state, 1, nSamples))
local best_state, best_energy = simaneal(initial_state, getNeighbour,
					 energy, temperature, 100000)
print(energy(best_state, 1, nSamples))

w:copy(best_state)
local output = model:forward(dataset.data)
err = criterion:forward(output, dataset.labels)
print(err)


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
print("test err", avg_err)
print("test acc (%)", 100 * n_good / dataset_test.data:size(1))