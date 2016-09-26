--Torch Cifar10
--Reference:  http://torch.ch/blog/2015/07/30/cifar.html
--https://github.com/szagoruyko/cifar.torch

-- Code for Wide Residual Networks http://arxiv.org/abs/1605.07146
-- (c) Sergey Zagoruyko, 2016
require 'xlua'
require 'optim'
require 'image'
require 'cunn'
local c = require 'trepl.colorize'
require 'dl4j-core-benchmark/src/main/java/org/deeplearning4j/Utils/torch-utils'
require 'logroll'
main_path='dl4j-core-benchmark/src/main/resources/torch-data'
dofile 'dl4j-core-benchmark/src/main/java/org/deeplearning4j/Cifar10/torch/provider.lua'

local tablex = require 'pl.tablex'

------------------------------------------------------------
-- Setup Variables

log = logroll.print_logger()
log.level = logroll.DEBUG

cmd = torch.CmdLine()
cmd:option('-gpu', false, 'boolean flag to use gpu for training')
cmd:option('-multi', false, 'boolean flag to use multi-gpu for training')
-- TODO go after res net which is closer to 95% accuracy
cmd:option('-model_type', 'mlp', 'Which model to run between nin and vgg')
config = cmd:parse(arg)


total_time = sys.clock()
torch.manualSeed(42)

opt = {
    gpu = config.gpu,
    multi = config.multi,
    -- TODO get file and labeled
    dataset = paths.concat(main_path, 'cifar10_whitened.t7'),
    save = 'logs',
    batchSize = 128,
    learningRateDecay = 1e-7,
    epoch_step = 25,
    max_epoch = 300,
    optimMethod = 'sgd',
    init_value = 10,
    nesterov = false,
    dropout = 0,
    imageSize = 32,
    cudnn = true,
    generate_graph = false,
    multiply_input_factor = 1,
    nGPU = 0,

--    channels = 1,
--    height = 32,
--    width = 32,
--    ninputs = 32*32,
}

if opt.multi then opt.nGPU = cutorch.getDeviceCount() end

optimState = {
    learningRate = 0.1,
    weightDecay = 5e-4,
}

classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
confusion = optim.ConfusionMatrix(classes)
opt = xlua.envparams(opt)

log.debug(opt)

------------------------------------------------------------
-- support functions

do -- data augmentation module
local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

function BatchFlip:__init()
    parent.__init(self)
    self.train = true
end

function BatchFlip:updateOutput(input)
    if self.train then
        local bs = input:size(1)
        local flip_mask = torch.randperm(bs):le(bs/2)
        for i=1,input:size(1) do
            if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
        end
    end
    self.output:set(input)
    return self.output
end
end

local function cast(t)
    require 'cunn'
    return t:cuda()
end


------------------------------------------------------------
-- Load data
data_load_time = sys.clock()
log.debug(c.blue '==>' ..' loading data')

verify_file = paths.concat(main_path,'provider')
if not paths.dirp(verify_file) then
    provider = Provider()
    print(provider)
    provider:normalize()
    torch.save(paths.concat(main_path,'provider.t7'), provider)
    log.debug(' Saved data at '..main_path.. 'provider.t7')
    paths.mkdir(paths.concat(main_path, 'provider')) -- Hacky avoid pulling data and processing
end
provider = torch.load(paths.concat(main_path,'provider.t7'))
provider.trainData.data = provider.trainData.data:float()
provider.testData.data = provider.testData.data:float()

testLogger = optim.Logger(paths.concat(main_path, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = false

data_load_time = sys.clock() - data_load_time

------------------------------------------------------------
-- Build model

function nin()
    local model = nn.Sequential()

    local function Block(...)
        local arg = {...}
        model:add(nn.SpatialConvolution(...))
        model:add(nn.SpatialBatchNormalization(arg[2],1e-3))
        model:add(nn.ReLU(true))
        return model
    end

    Block(3,192,5,5,1,1,2,2)
    Block(192,160,1,1)
    Block(160,96,1,1)
    model:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
    model:add(nn.Dropout())
    Block(96,192,5,5,1,1,2,2)
    Block(192,192,1,1)
    Block(192,192,1,1)
    model:add(nn.SpatialAveragePooling(3,3,2,2):ceil())
    model:add(nn.Dropout())
    Block(192,192,3,3,1,1,1,1)
    Block(192,192,1,1)
    Block(192,10,1,1)
    model:add(nn.SpatialAveragePooling(8,8,1,1):ceil())
    model:add(nn.View(10))

    for k,v in pairs(model:findModules(('%s.SpatialConvolution'):format(backend_name))) do
        v.weight:normal(0,0.05)
        v.bias:zero()
    end
    model:add(nn.SoftMax())
    return model
end

function vgg()
    local model = nn.Sequential()

    -- building block
    local function ConvBNReLU(nInputPlane, nOutputPlane)
        model:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
        model:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
        model:add(nn.ReLU(true))
        return model
    end
    -- Will use "ceil" MaxPooling because we want to save as much feature space as we can
    local MaxPooling = nn.SpatialMaxPooling

    ConvBNReLU(3,64):add(nn.Dropout(0.3))
    ConvBNReLU(64,64)
    model:add(MaxPooling(2,2,2,2):ceil())
    ConvBNReLU(64,128):add(nn.Dropout(0.4))
    ConvBNReLU(128,128)
    model:add(MaxPooling(2,2,2,2):ceil())
    ConvBNReLU(128,256):add(nn.Dropout(0.4))
    ConvBNReLU(256,256):add(nn.Dropout(0.4))
    ConvBNReLU(256,256)
    model:add(MaxPooling(2,2,2,2):ceil())
    ConvBNReLU(256,512):add(nn.Dropout(0.4))
    ConvBNReLU(512,512):add(nn.Dropout(0.4))
    ConvBNReLU(512,512)
    model:add(MaxPooling(2,2,2,2):ceil())
    ConvBNReLU(512,512):add(nn.Dropout(0.4))
    ConvBNReLU(512,512):add(nn.Dropout(0.4))
    ConvBNReLU(512,512)
    model:add(MaxPooling(2,2,2,2):ceil())
    model:add(nn.View(512))
    model:add(nn.Dropout(0.5))
    model:add(nn.Linear(512,512))
    model:add(nn.BatchNormalization(512))
    model:add(nn.ReLU(true))
    model:add(nn.Dropout(0.5))
    model:add(nn.Linear(512,10))

    -- initialization from MSR
    for k,v in pairs(model:findModules('nn.SpatialConvolution')) do
        local n = v.kW*v.kH*v.nOutputPlane
        v.weight:normal(0,math.sqrt(2/n))
        if v.bias then v.bias:zero() end
    end
    for k,v in pairs(model:findModules'nn.Linear') do
        v.bias:zero()
    end
    for i,v in ipairs(model:findModules'nn.SpatialConvolution') do
        v.bias = nil
        v.gradBias = nil
    end
    model:add(nn.SoftMax())
    return model
end


log.debug(c.blue '==>' ..' build model')
local model = nn.Sequential()
model:add(nn.BatchFlip():float())
model:add(cast(nn.Copy('torch.FloatTensor', torch.type(cast(torch.Tensor())))))
model:add(cast(dofile()))
if config.model_type == 'nin' then
    model:add(cast(dofile(nin())))
else
    model:add(cast(dofile(vgg())))
end

model:get(2).updateGradInput = function(input) return end
--model:add(nn.SoftMax()):cuda()
if opt.gpu then model = util.convertCuda(model, opt.nGPU) end
local parameters,gradParameters = model:getParameters()
opt.n_parameters = parameters:numel()
log.debug('Network has ', parameters:numel(), 'parameters')

local criterion = nn.CrossEntropyCriterion():cuda()
local optimState = tablex.deepcopy(opt)

log.debug(model)

------------------------------------------------------------
-- Train model
local f = function(inputs, targets)
    model:forward(inputs)
    local loss = criterion:forward(model.output, targets)
    local df_do = criterion:backward(model.output, targets)
    model:backward(inputs, df_do)
    return loss
end


function train(model)
    epoch = epoch or 1

    -- drop learning rate every "epoch_step" epochs
    if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end

    print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

    local targets = torch.CudaTensor(opt.batchSize)
    local indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
    -- remove last element so that all minibatches have equal size
    indices[#indices] = nil

    local loss = 0

    local tic = torch.tic()
    for t,v in ipairs(indices) do
        xlua.progress(t, #indices)

        local inputs = provider.trainData.data:index(1,v)
        targets:copy(provider.trainData.labels:index(1,v))

        optim[opt.optimMethod](function(x)
            if x ~= parameters then parameters:copy(x) end
            model:zeroGradParameters()
            loss = loss + f(inputs, targets)
            return f,gradParameters
        end, parameters, optimState)
    end
    confusion:batchAdd(outputs, targets)
    log.debug(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(tic)))
    train_acc = confusion.totalValid * 100
    confusion:zero()
    epoch = epoch + 1

    return loss / #indices
end
------------------------------------------------------------
-- Evaluate

function test(model)
    model:evaluate()
    print(c.blue '==>'.." evaluate model")
    local data_split = provider.testData.data:split(opt.batchSize,1)
    local labels_split = provider.testData.labels:split(opt.batchSize,1)

    confusion:batchAdd(model:forward(data_split), labels_split)

    confusion:updateValids()
    if opt.logger then print(confusion) end
    print('Accuracy: ', confusion.totalValid * 100)
    return confusion.totalValid * 100
end

------------------------------------------------------------
-- Run
local function timing(f, input) local s = torch.Timer(); return f(input), s:time().real end

log.debug(c.blue '==>' ..' train model')
model:training()
for epoch=1,opt.max_epoch do
    local loss, train_time = timing(train, model)
end

log.debug(c.blue '==>' ..' evaluate model')
local test_acc, test_time = timing(test, model)

torch.save(opt.save..'/model.t7', model:clearState())
total_time = sys.clock() - total_time

print("****************Example finished********************")
util.printTime('Data load', data_load_time)
util.printTime('Train', train_time)
util.printTime('Test', test_time)
util.printTime('Total', total_time)
