--
--The MIT License (MIT)
--
--Copyright (c) 2015 Sergey Zagoruyko
--
--Permission is hereby granted, free of charge, to any person obtaining a copy
--of this software and associated documentation files (the "Software"), to deal
--in the Software without restriction, including without limitation the rights
--to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
--copies of the Software, and to permit persons to whom the Software is
--furnished to do so, subject to the following conditions:
--
--The above copyright notice and this permission notice shall be included in all
--copies or substantial portions of the Software.
--
--THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
--IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
--FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
--AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
--LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
--OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
--SOFTWARE.
--
-- Reference: https://github.com/szagoruyko/cifar.torch

require 'nn'
require 'image'
require 'xlua'

local Provider = torch.class 'Provider'

function Provider:__init(full)
    local trsize = 50000
    local tesize = 10000
    local path_dataset = 'dl4j-core-benchmark/src/main/resources/torch-data'
    local verify_file = paths.concat(path_dataset, 'provider')
    -- download dataset
    if not paths.dirp(verify_file) then
        local remote = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'
        local tar = paths.concat(path_dataset, paths.basename(remote))
        os.execute('wget -cO' ..tar .. ' ' .. remote .. '; ' .. 'tar --strip-components=1 -zxvf ' .. tar .. ' -C ' .. path_dataset .. '; rm ' .. tar .. ';' )
    end

    -- load dataset
    self.trainData = {
        data = torch.Tensor(50000, 3072),
        labels = torch.Tensor(50000),
        size = function() return trsize end
    }
    local trainData = self.trainData
    for i = 0,4 do
        local subset = torch.load(paths.concat(path_dataset,'data_batch_' .. (i+1) .. '.t7'), 'ascii')
        trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
        trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
    end
    trainData.labels = trainData.labels + 1

    local subset = torch.load(paths.concat(path_dataset,'test_batch.t7'), 'ascii')
    self.testData = {
        data = subset.data:t():double(),
        labels = subset.labels[1]:double(),
        size = function() return tesize end
    }
    local testData = self.testData
    testData.labels = testData.labels + 1

    -- resize dataset (if using small version)
    trainData.data = trainData.data[{ {1,trsize} }]
    trainData.labels = trainData.labels[{ {1,trsize} }]

    testData.data = testData.data[{ {1,tesize} }]
    testData.labels = testData.labels[{ {1,tesize} }]

    -- reshape data
    trainData.data = trainData.data:reshape(trsize,3,32,32)
    testData.data = testData.data:reshape(tesize,3,32,32)
end

function Provider:normalize()
    ----------------------------------------------------------------------
    -- preprocess/normalize train/test sets
    --
    local trainData = self.trainData
    local testData = self.testData

    print '<trainer> preprocessing data (color space + normalization)'
    collectgarbage()

    -- preprocess trainSet
    local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
    for i = 1,trainData:size() do
        xlua.progress(i, trainData:size())
        -- rgb -> yuv
        local rgb = trainData.data[i]
        local yuv = image.rgb2yuv(rgb)
        -- normalize y locally:
        yuv[1] = normalization(yuv[{{1}}])
        trainData.data[i] = yuv
    end
    -- normalize u globally:
    local mean_u = trainData.data:select(2,2):mean()
    local std_u = trainData.data:select(2,2):std()
    trainData.data:select(2,2):add(-mean_u)
    trainData.data:select(2,2):div(std_u)
    -- normalize v globally:
    local mean_v = trainData.data:select(2,3):mean()
    local std_v = trainData.data:select(2,3):std()
    trainData.data:select(2,3):add(-mean_v)
    trainData.data:select(2,3):div(std_v)

    trainData.mean_u = mean_u
    trainData.std_u = std_u
    trainData.mean_v = mean_v
    trainData.std_v = std_v

    -- preprocess testSet
    for i = 1,testData:size() do
        xlua.progress(i, testData:size())
        -- rgb -> yuv
        local rgb = testData.data[i]
        local yuv = image.rgb2yuv(rgb)
        -- normalize y locally:
        yuv[{1}] = normalization(yuv[{{1}}])
        testData.data[i] = yuv
    end
    -- normalize u globally:
    testData.data:select(2,2):add(-mean_u)
    testData.data:select(2,2):div(std_u)
    -- normalize v globally:
    testData.data:select(2,3):add(-mean_v)
    testData.data:select(2,3):div(std_v)
end

