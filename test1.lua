require 'simaneal'
require 'nn'

local model = nn.Sequential()
model:add(nn.Linear(1, 2))
model:add(nn.Linear(2, 1))
local criterion = nn.MSECriterion()

local w, dw = model:getParameters()

local initial_state = w
local function getNeighbor(w)
   return w + torch.randn(w:size())
end
local function energy(p, i_step)
   w:copy(p)
   local input = torch.randn(1)
   local target = input
   local output = model:forward(input)
   local err = criterion:forward(output, target)
   return err
end
local function temperature(i_step)
   return 1/(1+i_step)
end

print(energy(initial_state, 1))
local best_state, best_energy = simaneal(initial_state, getNeighbor,
					 energy, temperature, 10000)
print(best_energy)