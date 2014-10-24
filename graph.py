import os
import subprocess
import matplotlib
import matplotlib.pyplot as pyplot
pyplot.switch_backend('qt4agg')

def getEnergies(n_hid):
    result = subprocess.Popen("""th -e "
require 'torch'
require 'paths'
require 'os'
require 'sys'
require 'string'

local filespath = '/archive/mfm352/annealing/outputs_ok_but_not_cvg/'

local function get_energies(n_hid)
   local filenames = {}
   for line in sys.ls(filespath):gmatch('(.-)\\n') do
      if line:sub(-4, -1) == '.t7b' then
	 if string.find(line, 'nhid-'..n_hid, 1, true) ~= nil then
	    if string.find(line, 'discret') ~= nil then
	       filenames[1+#filenames] = paths.concat(filespath, line)
            end
	 end
      end
   end

   local energies = torch.Tensor(#filenames)
   for i, filename in pairs(filenames) do
      local file = torch.load(filename)
      energies[i] = file.testing_acc
   end
   return energies
end

local energies = get_energies(%d)

for i = 1, energies:size(1) do
   io.write(energies[i] .. '\\n')
end
io.flush()
" """%(n_hid), shell=True, stdout = subprocess.PIPE)
    energies = [float(x.strip()) for x in result.stdout.readlines() if x.strip() != '']
    return energies

#energies_5 = getEnergies(5)
#energies_10 = getEnergies(10)
#energies_15 = getEnergies(15)
energies_20 = getEnergies(20)
#energies_25 = getEnergies(25)
nbins = 100
xmin = 0.
xmax = 1.
print(energies_20)
fig = pyplot.figure()
#pyplot.hist(energies_5, nbins, (xmin, xmax), label = u'n=5')
#pyplot.hist(energies_10, nbins, (xmin, xmax), label = u'n=10')
#pyplot.hist(energies_15, nbins, (xmin, xmax), label = u'n=15')
pyplot.hist(energies_20, nbins, (xmin, xmax), label = u'n=20')
#pyplot.hist(energies_25, nbins, (xmin, xmax), label = u'n=25')
pyplot.legend()
pyplot.show()
fig.savefig('results.pdf')
