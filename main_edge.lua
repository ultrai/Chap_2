require 'sys'
require 'torch'
require 'cunn'
require 'nn' 
require 'cudnn'
--matio = require 'matio'
require 'optim'
require 'cutorch'
require 'math'
im = require 'image'
require 'nngraph'
cutorch.setDevice(1)
torch.setdefaulttensortype('torch.FloatTensor')

--local MSRInit = require './initialization.lua'

local MaxPooling = cudnn.SpatialMaxPooling
local Convolution = cudnn.SpatialConvolution
local BatchNorm = cudnn.SpatialBatchNormalization
local UpConvolution = nn.SpatialFullConvolution
local Join = nn.JoinTable
local CAdd = nn.CAddTable
local ReLU = nn.ReLU
local Dropout = nn.Dropout

--- Creates a conv layer given number of input feature maps and number of output feature maps, with dropout
-- @param nIn Number of input feature maps
-- @param nOut Number of output feature maps
-- @param dropout Dropout layer if required
local function ConvLayers(nIn, nOut, dropout)
	local kW, kH, dW, dH, padW, padH = 5, 5, 1, 1, 2, 2 -- 3, 3, 1, 1, 1, 1 -- parameters for 'same' conv layers
	local net = nn.Sequential()
	net:add(Convolution(nIn, nOut, kW, kH, dW, dH, padW, padH))
	net:add(BatchNorm(nOut))
	net:add(ReLU(true))
	if dropout then net:add(Dropout(dropout)) end
	net:add(Convolution(nOut, nOut, kW, kH, dW, dH, padW, padH))
	net:add(BatchNorm(nOut))
	net:add(ReLU(true))
	if dropout then net:add(Dropout(dropout)) end
	return net
end

--- Returns model, name which is used for the naming of models generated while training
local function createModel(opt)
	opt = opt or {}
	local nbClasses = opt.nbClasses or 9 	-- # of labls
	local nbChannels = opt.nbChannels or 1 	-- # of labls
        local nfilt = 32
        local grow = {1,2,4,8,16}
	local input = nn.Identity()()
         
	local D1 = ConvLayers(nbChannels,nfilt*grow[1])(input)
	local D2 = ConvLayers(nfilt*grow[1],nfilt*grow[2])(MaxPooling(2,2)(D1))
	local D3 = ConvLayers(nfilt*grow[2],nfilt*grow[3])(MaxPooling(2,2)(D2))
	local D4 = ConvLayers(nfilt*grow[3],nfilt*grow[4])(MaxPooling(2,2)(D3))

	local B = ConvLayers(nfilt*grow[4],nfilt*grow[5])(MaxPooling(2,2)(D4))

	local U4 = ConvLayers(nfilt*grow[5],nfilt*grow[4])(Join(1,3)({ D4, ReLU(true)(UpConvolution(nfilt*grow[5],nfilt*grow[4], 2,2,2,2)(B)) }))
	local U3 = ConvLayers(nfilt*grow[4],nfilt*grow[3])(Join(1,3)({ D3, ReLU(true)(UpConvolution(nfilt*grow[4],nfilt*grow[3], 2,2,2,2)(U4)) }))
	local U2 = ConvLayers(nfilt*grow[3],nfilt*grow[2])(Join(1,3)({ D2, ReLU(true)(UpConvolution(nfilt*grow[3],nfilt*grow[2], 2,2,2,2)(U3)) }))
	local U1 = ConvLayers(nfilt*grow[2],nfilt*grow[1])(Join(1,3)({ D1, ReLU(true)(UpConvolution(nfilt*grow[2],nfilt*grow[1], 2,2,2,2)(U2))   }))
  local U = Convolution(nfilt*grow[1], nbClasses, 1,1)(U1)

	local U4_edge = ConvLayers(nfilt*grow[1]+1,nfilt*grow[1])(Join(1,3)({ input, ReLU(true)(UpConvolution(nfilt*grow[5],nfilt*grow[1], 16,16,16,16)(B)) }))
  local U3_edge = ConvLayers(2*nfilt*grow[1],nfilt*grow[1])(Join(1,3)({ U4_edge, ReLU(true)(UpConvolution(nfilt*grow[4],nfilt*grow[1], 8,8,8,8)(U4)) }))
	local U2_edge = ConvLayers(2*nfilt*grow[1],nfilt*grow[1])(Join(1,3)({ U3_edge, ReLU(true)(UpConvolution(nfilt*grow[3],nfilt*grow[1], 4,4,4,4)(U3)) }))
	local U1_edge = ConvLayers(2*nfilt*grow[1],1)(Join(1,3)({ U2_edge, ReLU(true)(UpConvolution(nfilt*grow[2],nfilt*grow[1], 2,2,2,2)(U2))   }))

	local net = nn.Sequential()
	net:add(nn.gModule({input}, {U,U1_edge}))
	return net
end

model = createModel()
model = torch.load('Model_unet')
model:cuda()
criteria1 = cudnn.SpatialCrossEntropyCriterion() 
criteria1:cuda()
criteria2 = nn.MSECriterion() 
criteria2:cuda()

criterion = nn.ParallelCriterion():add(criteria1,1):add(criteria2,1)
criterion:cuda()

--kk = model:forward(torch.Tensor(1,1,64,64):cuda())

require 'hdf5'
myFile = hdf5.open('train.hdf5', 'r')
Temp = myFile:read(''):all()
myFile:close()
temp=Temp.data
temp = temp:type('torch.FloatTensor')
X_train = torch.Tensor(temp:size(1),temp:size(2),temp:size(3),temp:size(4)):copy(temp)
temp=Temp.label
temp = temp:type('torch.FloatTensor')
Y_train = torch.Tensor(temp:size(1),temp:size(2),temp:size(3)):copy(temp):add(1)
temp=Temp.label2
temp = temp:type('torch.FloatTensor')
Y_train2 = torch.ge(torch.Tensor(temp:size(1),temp:size(2),temp:size(3)):copy(temp),1)

--Y_train = Y_train:reshape(Y_train:size(1),Y_train:size(3),Y_train:size(4))
Temp = nil
temp=nil
--kk =Y_train[{{1,10},{},{}}],Y_train2[{{1,10},{},{}}]



torch.manualSeed(27)

Cost = 999999
eta = 0
func = function(x)
        if x ~= parameters then
              parameters:copy(x)
        end
        gradParameters:zero()
        f = 0
if neval <5400 then
        f_test = 0
end
        neval = neval + 1
if (neval%540) ==0 then
state.learningRate = state.learningRate/10
--state.weightDecay = state.weightDecay/2.5
end
             output = model:forward(inputs:cuda())
            f= criterion:forward(output,{targets:cuda(),targets2:cuda()})
            df_do = criterion:backward(output,{targets:cuda(),targets2:cuda()})
            model:backward(inputs:cuda(), df_do)
 	    collectgarbage()
        table.insert(train,f/inputs:size(1))
if Cost>f then
Cost = f
eta = neval
end

if (neval%5400000000==0) then     
        model:evaluate()

 out_train = torch.zeros(inputs_test:size(1),inputs_train:size(3),inputs_test:size(4))
        for i = 1,inputs_test:size(1) do
            output = model:forward(inputs_train[{{i},{},{},{}}]:cuda())
            oo,out_train[{{i},{},{}}] = torch.max(output[1]:float(),1)
            collectgarbage()
        end
 
f_test = torch.sum(torch.eq(out_train, targets_train))/(496*512)
        table.insert(test,f_test/inputs_train:size(1))
model:training()
end
        print(string.format('after %d evaluations J(x) = %f took %f %f %f', neval, f,  sys:toc(), Cost,f_test))
      return f/inputs:size(1),gradParameters:div(inputs:size(1))
end

optimState = {maxIter = 100}
state = {
--maxIter = 100,
learningRate = 1e-4,
   momentum = 0.9,
--dampening = 0,
--weightDecay = 1e-5,
--nesterov = true,
}
optimMethod = optim.adam--adadelta--adam--cg--adagrad--adam
sys:tic()
train = {}
test = {}
neval = 0
batch =32

f_test =0
for epoch = 1,300 do
    for temp = 1,X_train:size(1)-batch,batch do
        inputs = X_train[{{temp,temp+batch},{},{},{}}]
        targets = Y_train[{{temp,temp+batch},{},{}}]
        targets2 = Y_train2[{{temp,temp+batch},{},{}}]
        
        parameters,gradParameters = model:getParameters()
        optimMethod(func, parameters,state)
    end
    inputs = X_train[{{X_train:size(1)-batch,X_train:size(1)},{},{},{}}]:cuda()
targets = Y_train[{{X_train:size(1)-batch,X_train:size(1)},{},{}}]:cuda()
targets2 = Y_train2[{{X_train:size(1)-batch,X_train:size(1)},{},{}}]:cuda()

parameters,gradParameters = model:getParameters()
        
optimMethod(func, parameters,state)
if epoch%50==0 then
state.learningRate = 1e-5
end
print(f_test/X_train:size(1))
torch.save('Model_unet',model:clearState())

end
train = torch.Tensor(train)
test = torch.Tensor(test)
torch.save('Model_unet',model)
--torch.save('train',train)--torch.save('train.txt',train,'ascii')
--torch.save('test',test)

