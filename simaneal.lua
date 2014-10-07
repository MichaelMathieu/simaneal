require 'torch'
require 'math'

function simaneal(initial_state, getNeighbour, energy, temperature, n_steps)
   local cur_state = initial_state
   local cur_energy = energy(cur_state, 1)

   local best_state = cur_state
   local best_energy = cur_energy

   local function P(cur_energy, new_energy, T)
      if new_energy < cur_energy then
	 return 1.0
      else
	 return math.exp((cur_energy - new_energy) / T)
      end
   end
   
   for i_step = 1, n_steps do
      local T = temperature(i_step)
      local new_state = getNeighbour(cur_state)
      local new_energy = energy(new_state, i_step)
      if P(cur_energy, new_energy, T) > torch.uniform() then
	 cur_state = new_state
	 cur_energy = new_energy
	 print("new", i_step, cur_energy)
	 if cur_energy < best_energy then
	    best_state = cur_state
	    best_energy = cur_energy
	 end
      end
   end

   return best_state, best_energy
end
      
      