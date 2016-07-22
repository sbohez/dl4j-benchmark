--Torch Cifar10
--Reference: http://torch.ch/blog/2015/07/30/cifar.html

require 'torch'
require 'nn'
require 'optim'
require 'paths'

torch.manualSeed(42)

opt = {
    gpu = true,
    max_epoch = 15,
    numExamples = 59904, -- after it throws errors in target and doesn't properly load
    numTestExamples = 10000,
    batchSize = 128,
    noutputs = 10,
    channels = 1,
    height = 32,
    width = 32,
    --    ninputs = 32*32,

    --    height = 28,
    --    width = 28,
    ninputs = 28*28,
    nhidden = 1000,
    coefL2 = 1e-4,
    path_dataset = 'src/main/resources/torch-data'

}


optimState = {
    learningRate = 0.006,
    weightDecay = 1e-4,
    nesterov = true,
    momentum =  0.9,
    dampening = 0
}

classes = {'1','2','3','4','5','6','7','8','9','10'}
geometry = {opt.height,opt.width}

------------------------------------------------------------
-- print('Load data')


provider = Provider()
provider:normalize()
torch.save(paths.concat(opt.path_dataset, 'provider.t7'),provider)

------------------------------------------------------------
-- print('Build model')
function very_deep_model()

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
    local function MSRinit(net)
        local function init(name)
            for k,v in pairs(net:findModules(name)) do
                local n = v.kW*v.kH*v.nOutputPlane
                v.weight:normal(0,math.sqrt(2/n))
                v.bias:zero()
            end
        end
        init'nn.SpatialConvolution'
    end
    MSRinit(vgg)
    return vgg
end

------------------------------------------------------------
-- print('Train model')
function training()
    local MAX_EPOCH = 20
    local x = torch.load(string.format("%s/train_x.bin", DATA_DIR))
    local y = torch.load(string.format("%s/train_y.bin", DATA_DIR))

    local model = very_deep_model() --:cuda()
    local criterion = nn.MSECriterion()--:cuda()

    local sgd_config = {
        learningRate = 1.0,
        learningRateDecay = 5.0e-6,
        momentum = 0.9,
        xBatchSize = 64
    }
    local params = nil

    print("data augmentation ..")
    x, y = data_augmentation(x, y)
    collectgarbage()

    print("preprocessing ..")
    params = preprocessing(x)
    torch.save("models/preprocessing_params.bin", params)
    collectgarbage()

    for epoch = 1, MAX_EPOCH do
        print("# " .. epoch)
        if epoch == MAX_EPOCH then
            -- final epoch
            sgd_config.learningRateDecay = 0
            sgd_config.learningRate = 0.01
        end
        model:training()
        print(minibatch_sgd(model, criterion, x, y,
            CLASSES, sgd_config))
        model:evaluate()
        torch.save(string.format("models/very_deep_%d.model", epoch), model)
        epoch = epoch + 1

        collectgarbage()
    end
    return model
end
------------------------------------------------------------
-- print('Evaluate')

local function test(model, params, test_x, test_y, classes)
    local confusion = optim.ConfusionMatrix(classes)
    for i = 1, test_x:size(1) do
        local preds = torch.Tensor(10):zero()
        local x = data_augmentation(test_x[i])
        local step = 64
        preprocessing(x, params)
        for j = 1, x:size(1), step do
            local batch = torch.Tensor(step, x:size(2), x:size(3), x:size(4)):zero()
            local n = step
            if j + n > x:size(1) then
                n = 1 + n - ((j + n) - x:size(1))
            end
            batch:narrow(1, 1, n):copy(x:narrow(1, j, n))
            local z = model:forward(batch:cuda()):float()
            -- averaging
            for k = 1, n do
                preds = preds + z[k]
            end
        end
        preds:div(x:size(1))
        confusion:add(preds, test_y[i])
        xlua.progress(i, test_x:size(1))
    end
    xlua.progress(test_x:size(1), test_x:size(1))
    return confusion
end

training()
test()
