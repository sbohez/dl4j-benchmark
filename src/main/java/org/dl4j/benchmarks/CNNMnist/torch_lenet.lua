--
--
-- Note documentation harder to explore esp for newbies
--

require 'torch'
require 'nn'
require 'optim'
require 'mnist'

torch.setdefaulttensortype('torch.FloatTensor')

-- epoch tracker
opt = {
    gpu = true,
    max_epoch = 11,
    numExamples = 60000 , -- numExamples
    numTestExamples = 10000,
    batchSize = 66,
    testBatchSize = 100,
    noutputs = 10,
    channels = 1,
    height = 28,
    width = 28,
    ninputs = 784,
    coefL2 = 5e-4
}

optimState = {
    learningRate = 1e-2,
    weightDecay = opt.coefL2,
    momentum =  0.9

}

classes = {'1','2','3','4','5','6','7','8','9','10'}

------------------------------------------------------------
-- print('Load data')
-- create training set and normalize
trainData = mnist.loadTrainSet(opt.numExamples, geometry)
mean = trainData.data:mean()
std =  trainData.data:std()
trainData:normalizeGlobal(mean, std)

-- create test set and normalize
testData = mnist.loadTestSet(opt.numTestExamples, geometry)
mean = testData.data:mean()
std =  testData.data:std()
testData:normalizeGlobal(mean, std)

------------------------------------------------------------
-- print('Build model')
model = nn.Sequential()
-- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
model:add(nn.SpatialConvolutionMM(1, 20, 5, 5))
model:add(nn.Identity())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
-- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
model:add(nn.SpatialConvolutionMM(20, 50, 5, 5))
model:add(nn.Identity())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
-- stage 3 : standard 2-layer MLP:
model:add(nn.Reshape(50*4*4))
model:add(nn.Linear(50*4*4, 500))
model:add(nn.Relu())
model:add(nn.Linear(500, #classes))

--flattens & creates views for optim to process param and gradients
parameters,gradParameters = model:getParameters()

model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

--print(model)

------------------------------------------------------------
-- print('Train model')
function train(dataset)
    -- epoch tracker
    epoch = epoch or 1
    -- set model to training mode (for modules that differ in training and testing, like Dropout)
    model:training()

--    loops from 1 to full dataset size by batchsize
    for t = 1,opt.numExamples,opt.batchSize do
        -- create mini batch
        local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
        local targets = torch.Tensor(opt.batchSize)
        local k = 1
        for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
            -- load new sample
            local sample = dataset[i]
            local input = sample[1]:clone()
            local _,target = sample[2]:clone():max(1)
            target = target:squeeze()
            inputs[k] = input
            targets[k] = target
            k = k + 1
        end

        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
            -- just in case:
            collectgarbage()

            -- get new parameters
            if x ~= parameters then
                parameters:copy(x)
            end

            -- reset gradients
            gradParameters:zero()

            -- evaluate function for complete mini batch
            local outputs = model:forward(inputs)
            local f = criterion:forward(outputs, targets)

            -- estimate df/dW
            local df_do = criterion:backward(outputs, targets)
            model:backward(inputs, df_do)

            -- penalties (L1 and L2):
            local norm= torch.norm
            -- Loss:
            f = f + opt.coefL2 * norm(parameters,2)^2/2
            -- Gradients:
            gradParameters:add(parameters:clone():mul(opt.coefL2))

            -- return f and df/dX
            return f, gradParameters
        end

        optim.nesterov(feval, parameters, optimState)
    end

    -- next epoch
    epoch = epoch + 1
end

------------------------------------------------------------
-- print('Evaluate')
-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

function test(dataset)

    -- test over given dataset
    for t = 1,dataset:size(),opt.testBatchSize do
        -- disp progress
        xlua.progress(t, dataset:size())

        -- create mini batch
        local inputs = torch.Tensor(opt.testBatchSize,1,geometry[1],geometry[2])
        local targets = torch.Tensor(opt.testBatchSize)
        local k = 1
        for i = t,math.min(t+opt.testBatchSize-1,opt.numTestExamples) do
            -- load new sample
            local sample = dataset[i]
            local input = sample[1]:clone()
            local _,target = sample[2]:clone():max(1)
            target = target:squeeze()
            inputs[k] = input
            targets[k] = target
            k = k + 1
        end

        -- test samples
        local preds = model:forward(inputs)

        -- confusion:
        for i = 1,opt.batchSize do
            confusion:add(preds[i], targets[i])
        end
    end

    -- print confusion matrix
    print(confusion)
    testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
    confusion:zero()
end


for _ = 1,opt.max_epoch do
    local time = sys.clock()
    train(trainData)
    test(testData)
    time = sys.clock() - time
    print("Total time: " .. sys.clock() - time)
end