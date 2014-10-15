require 'paths'
require 'os'
require 'sys'
require 'xlua'
require 'gnuplot'
require 'string'

local filespath = 'outputs'

local function get_energies(n_hid)
   local filenames = {}
   for line in sys.ls(filespath):gmatch("(.-)\n") do
      if line:sub(-4, -1) == '.t7b' then
	 if string.find(line, 'nhid-'..n_hid, 1, true) ~= nil then
	    filenames[1+#filenames] = paths.concat(filespath, line)
	 end
      end
   end

   local energies = torch.Tensor(#filenames)
   for i, filename in pairs(filenames) do
      xlua.progress(i, #filenames)
      local file = torch.load(filename)
      energies[i] = file.testing_acc
   end
   return energies
end

local energies_5 = get_energies(5)
local energies_10 = get_energies(10)

gnuplot.hist(energies_5, 20)
gnuplot.hist(energies_10, 20)