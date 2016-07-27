--Torch Cifar10
--Reference:  http://torch.ch/blog/2015/07/30/cifar.html
--https://github.com/szagoruyko/cifar.torch

-- Code for Wide Residual Networks http://arxiv.org/abs/1605.07146
-- (c) Sergey Zagoruyko, 2016
require 'xlua'
require 'optim'
require 'image'
require 'cunn'
require 'cudnn'
local c = require 'trepl.colorize'
local json = require 'cjson'
require 'src/main/java/org/dl4j/benchmarks/Utils/benchmark-util'

-- for memory optimizations and graph generation
local optnet = require 'optnet'
local graphgen = require 'optnet.graphgen'
local iterm = require 'iterm'
require 'iterm.dot'

total_time = sys.clock()
torch.manualSeed(42)

opt = {
    -- TODO get file and labeled
    dataset = 'src/main/resources/torch-data/cifar10_whitened.t7',
    save = 'logs',
    batchSize = 128,
    learningRate = 0.1,
    learningRateDecay = 0,
    learningRateDecayRatio = 0.2,
    weightDecay = 0.0005,
    dampening = 0,
    momentum = 0.9,
    epoch_step = "80",
    max_epoch = 300,
    model = 'nin',
    optimMethod = 'sgd',
    init_value = 10,
    depth = 50,
    shortcutType = 'A',
    nesterov = false,
    dropout = 0,
    hflip = true,
    randomcrop = 4,
    imageSize = 32,
    randomcrop_type = 'zero',
    cudnn_fastest = true,
    cudnn_deterministic = false,
    optnet_optimize = true,
    generate_graph = false,
    multiply_input_factor = 1,
    widen_factor = 1,
    nGPU = 0,
--    gpu = true,
--    channels = 1,
--    height = 32,
--    width = 32,
--    ninputs = 32*32,
--    coefL2 = 1e-4,

}

opt = xlua.envparams(opt)

opt.epoch_step = tonumber(opt.epoch_step) or loadstring('return '..opt.epoch_step)()
print(opt)

------------------------------------------------------------
-- support functions

do -- random crop
local RandomCrop, parent = torch.class('nn.RandomCrop', 'nn.Module')

function RandomCrop:__init(pad, mode)
    assert(pad)
    parent.__init(self)
    self.pad = pad
    if mode == 'reflection' then
        self.module = nn.SpatialReflectionPadding(pad,pad,pad,pad)
    elseif mode == 'zero' then
        self.module = nn.SpatialZeroPadding(pad,pad,pad,pad)
    else
        error'unknown mode'
    end
    self.train = true
end

function RandomCrop:updateOutput(input)
    assert(input:dim() == 4)
    local imsize = input:size(4)
    if self.train then
        local padded = self.module:forward(input)
        local x = torch.random(1,self.pad*2 + 1)
        local y = torch.random(1,self.pad*2 + 1)
        self.output = padded:narrow(4,x,imsize):narrow(3,y,imsize)
    else
        self.output:set(input)
    end
    return self.output
end

function RandomCrop:type(type)
    self.module:type(type)
    return parent.type(self, type)
end
end

do -- random horizontal flip
local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

function BatchFlip:updateOutput(input)
    self.train = self.train == nil and true or self.train
    if self.train then
        local bs = input:size(1)
        local flip_mask = torch.randperm(bs):le(bs/2)
        for i=1,input:size(1) do
            if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
        end
    end
    self.output:resize(input:size()):copy(input)
    return self.output
end
end
------------------------------------------------------------
-- print('Load data')
data_load_time = sys.clock()
print(c.blue '==>' ..' loading data')
local provider = torch.load(opt.dataset)
opt.num_classes = provider.testData.labels:max()
data_load_time = sys.clock() - data_load_time

------------------------------------------------------------
-- print('Build model')
print(c.blue '==>' ..' loading data')
function vgg()
    local vgg = nn.Sequential()
    -- building block
    local function ConvBNReLU(nInputPlane, nOutputPlane)
        vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
        vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
        vgg:add(nn.ReLU(true))
        return vgg
    end
    -- Will use "ceil" MaxPooling because we want to save as much feature space as we can
    local MaxPooling = nn.SpatialMaxPooling

    ConvBNReLU(3,64):add(nn.Dropout(0.3))
    ConvBNReLU(64,64)
    vgg:add(MaxPooling(2,2,2,2):ceil())
    ConvBNReLU(64,128):add(nn.Dropout(0.4))
    ConvBNReLU(128,128)
    vgg:add(MaxPooling(2,2,2,2):ceil())
    ConvBNReLU(128,256):add(nn.Dropout(0.4))
    ConvBNReLU(256,256):add(nn.Dropout(0.4))
    ConvBNReLU(256,256)
    vgg:add(MaxPooling(2,2,2,2):ceil())
    ConvBNReLU(256,512):add(nn.Dropout(0.4))
    ConvBNReLU(512,512):add(nn.Dropout(0.4))
    ConvBNReLU(512,512)
    vgg:add(MaxPooling(2,2,2,2):ceil())
    ConvBNReLU(512,512):add(nn.Dropout(0.4))
    ConvBNReLU(512,512):add(nn.Dropout(0.4))
    ConvBNReLU(512,512)
    vgg:add(MaxPooling(2,2,2,2):ceil())
    vgg:add(nn.View(512))
    vgg:add(nn.Dropout(0.5))
    vgg:add(nn.Linear(512,512))
    vgg:add(nn.BatchNormalization(512))
    vgg:add(nn.ReLU(true))
    vgg:add(nn.Dropout(0.5))
    vgg:add(nn.Linear(512,10))

    -- initialization from MSR
    for k,v in pairs(model:findModules('nn.SpatialConvolution')) do
        local n = v.kW*v.kH*v.nOutputPlane
        v.weight:normal(0,math.sqrt(2/n))
        if v.bias then v.bias:zero() end
    end
    for k,v in pairs(vgg:findModules'nn.Linear') do
        v.bias:zero()
    end
    for i,v in ipairs(model:findModules'nn.SpatialConvolution') do
        v.bias = nil
        v.gradBias = nil
    end
    return vgg
end

print(c.blue '==>' ..' configuring model')
local model = nn.Sequential()
local net = vgg():cuda()
do
    function nn.Copy.updateGradInput() end
    local function add(flag, module) if flag then model:add(module) end end
    add(opt.hflip, BatchFlip():float())
    add(opt.randomcrop > 0, RandomCrop(opt.randomcrop, opt.randomcrop_type):float())
    model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
    add(opt.multiply_input_factor ~= 1, nn.MulConstant(opt.multiply_input_factor):cuda())

    cudnn.convert(net, cudnn)
    cudnn.benchmark = true
    if opt.cudnn_fastest then
        for i,v in ipairs(net:findModules'cudnn.SpatialConvolution') do v:fastest() end
    end
    if opt.cudnn_deterministic then
        net:apply(function(m) if m.setMode then m:setMode(1,1,1) end end)
    end

    print(net)
    print('Network has', #net:findModules'cudnn.SpatialConvolution', 'convolutions')

    local sample_input = torch.randn(8,3,opt.imageSize,opt.imageSize):cuda()
    if opt.generate_graph then
        iterm.dot(graphgen(net, sample_input), opt.save..'/graph.pdf')
    end
    if opt.optnet_optimize then
        optnet.optimizeMemory(net, sample_input, {inplace = false, mode = 'training'})
    end

    model:add(makeDataParallelTable(net, opt.nGPU))
end

local function log(t) print('json_stats: '..json.encode(tablex.merge(t,opt,true))) end

print('Will save at '..opt.save)
paths.mkdir(opt.save)

local parameters,gradParameters = model:getParameters()

opt.n_parameters = parameters:numel()
print('Network has ', parameters:numel(), 'parameters')

print(c.blue'==>' ..' setting criterion')
local criterion = nn.CrossEntropyCriterion():cuda()

-- a-la autograd
local f = function(inputs, targets)
    model:forward(inputs)
    local loss = criterion:forward(model.output, targets)
    local df_do = criterion:backward(model.output, targets)
    model:backward(inputs, df_do)
    return loss
end

print(c.blue'==>' ..' configuring optimizer')
local optimState = tablex.deepcopy(opt)


------------------------------------------------------------
-- print('Train model')
function train()
    model:training()

    local targets = torch.CudaTensor(opt.batchSize)
    local indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
    -- remove last element so that all minibatches have equal size
    indices[#indices] = nil

    local loss = 0

    for t,v in ipairs(indices) do
        local inputs = provider.trainData.data:index(1,v)
        targets:copy(provider.trainData.labels:index(1,v))

        optim[opt.optimMethod](function(x)
            if x ~= parameters then parameters:copy(x) end
            model:zeroGradParameters()
            loss = loss + f(inputs, targets)
            return f,gradParameters
        end, parameters, optimState)
    end

    return loss / #indices
end
------------------------------------------------------------
-- print('Evaluate')
confusion = optim.ConfusionMatrix(classes)
function test()
    model:evaluate()
    local confusion = optim.ConfusionMatrix(opt.num_classes)
    local data_split = provider.testData.data:split(opt.batchSize,1)
    local labels_split = provider.testData.labels:split(opt.batchSize,1)

    for i,v in ipairs(data_split) do
        confusion:batchAdd(model:forward(v), labels_split[i])
    end

    confusion:updateValids()
    return confusion.totalValid * 100
end

------------------------------------------------------------
for epoch=1,opt.max_epoch do
    print('==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    -- drop learning rate and reset momentum vector
    if torch.type(opt.epoch_step) == 'number' and epoch % opt.epoch_step == 0 or
            torch.type(opt.epoch_step) == 'table' and tablex.find(opt.epoch_step, epoch) then
        opt.learningRate = opt.learningRate * opt.learningRateDecayRatio
        optimState = tablex.deepcopy(opt)
    end

    local function t(f) local s = torch.Timer(); return f(), s:time().real end

    local loss, train_time = t(train)
    local test_acc, test_time = t(test)

    log{
        loss = loss,
        epoch = epoch,
        test_acc = test_acc,
        lr = opt.learningRate,
        train_time = train_time,
        test_time = test_time,
    }
end

torch.save(opt.save..'/model.t7', net:clearState())
total_time = sys.clock() - total_time

print("****************Example finished********************")
util.printTime('Data load', data_load_time)
util.printTime('Train', train_time)
util.printTime('Test', test_time)
util.printTime('Total', total_time)
