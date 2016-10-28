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
	--net:add(Convolution(nIn, nOut, kW, kH, dW, dH, padW, padH))
  net:add(nn.SpatialReflectionPadding(2*padW, 2*padH, 2*padW, 2*padH)) 
  net:add(Convolution(nIn, nOut, kW, kH))
	net:add(BatchNorm(nOut))
	net:add(ReLU(true))
	if dropout then net:add(Dropout(dropout)) end
	--net:add(Convolution(nOut, nOut, kW, kH, dW, dH, padW, padH))
  net:add(Convolution(nOut, nOut, kW, kH))
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

	local net = nn.Sequential()
	net:add(nn.gModule({input}, {U}))
	return net
end

model = createModel()
--model = torch.load('Model_unet')
model:cuda()
criterion = cudnn.SpatialCrossEntropyCriterion() 
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
Y_train = torch.Tensor(temp:size(1),temp:size(3),temp:size(4)):copy(temp):add(1)
--temp=Temp.label2
--temp = temp:type('torch.FloatTensor')
--Y_train2 = torch.ge(torch.Tensor(temp:size(1),temp:size(3),temp:size(4)):copy(temp),1)

Temp = nil
temp=nil
myFile = hdf5.open('test.hdf5', 'r')
Temp = myFile:read(''):all()
myFile:close()
temp=Temp.data
temp = temp:type('torch.FloatTensor')
X_test = torch.Tensor(temp:size(1),temp:size(2),temp:size(3),temp:size(4)):copy(temp)
temp=Temp.label
temp = temp:type('torch.FloatTensor')
Y_test = torch.Tensor(temp:size(1),temp:size(2),temp:size(3)):copy(temp):add(1)


--Y_train = Y_train:reshape(Y_train:size(1),Y_train:size(3),Y_train:size(4))
Temp = nil
temp=nil
--kk =Y_train[{{1,10},{},{}}],Y_train2[{{1,10},{},{}}]



torch.manualSeed(27)

Cost = 999999
eta = 0
f_test = 0
func = function(x)
        if x ~= parameters then
              parameters:copy(x)
        end
        gradParameters:zero()
        
--if neval <5400 then
--        f_test = 0
--end
        neval = neval + 1
--if (neval%540) ==0 then
--state.learningRate = state.learningRate/10
--state.weightDecay = state.weightDecay/2.5
--end
             output = model:forward(inputs:cuda())
            f= criterion:forward(output,targets:cuda())
            df_do = criterion:backward(output,targets:cuda())
            model:backward(inputs:cuda(), df_do)
 	    collectgarbage()
        table.insert(train,f/inputs:size(1))
if Cost>f then
Cost = f
eta = neval
end

if (ee==1) then     
    model:evaluate()
    out_train = torch.zeros(X_test:size(1),X_train:size(3),X_train:size(4))
    batch2 =32
    for i = 1,X_test:size(1)-batch2,batch2 do
        output = model:forward(X_test[{{i,i+batch2},{},{},{}}]:cuda())
        oo,out_train[{{i,i+batch2},{},{}}] = torch.max(output:float(),1)
        collectgarbage()
    end
    output = model:forward(X_test[{{X_test:size(1)-batch2,X_test:size(1)},{},{},{}}]:cuda())
    oo,out_train[{{X_test:size(1)-batch2,X_test:size(1)},{},{}}] = torch.max(output:float(),1)
    f_test = torch.mean(torch.eq(out_train, Y_test))
    table.insert(test,f_test)
model:training()
end
        
        print(string.format('after %d evaluations J(x) = %f took %f with minimum cost %f and current test error %f', neval, f,  sys:toc(), Cost,f_test))
      return f,gradParameters
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
batch =512+256

f_test =0
for epoch = 1,10 do
  ee=0
    for temp = 1,X_train:size(1)-batch,batch do
        inputs = X_train[{{temp,temp+batch},{},{},{}}]
        targets = Y_train[{{temp,temp+batch},{},{}}]
        --targets2 = Y_train2[{{temp,temp+batch},{},{}}]
        
        parameters,gradParameters = model:getParameters()
        optimMethod(func, parameters,state)
    end
    ee=1
    inputs = X_train[{{X_train:size(1)-batch,X_train:size(1)},{},{},{}}]:cuda()
targets = Y_train[{{X_train:size(1)-batch,X_train:size(1)},{},{}}]:cuda()
--targets2 = Y_train2[{{X_train:size(1)-batch,X_train:size(1)},{},{}}]:cuda()

parameters,gradParameters = model:getParameters()
        
optimMethod(func, parameters,state)
if epoch%20==0 then
state.learningRate = state.learningRate/10
end
print(f_test/X_train:size(1))
torch.save('Model_unet',model:clearState())

end
train = torch.Tensor(train)
test = torch.Tensor(test)
torch.save('Model_unet',model)
--torch.save('train',train)--torch.save('train.txt',train,'ascii')
--torch.save('test',test)

