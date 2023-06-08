/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.pytorch.engine;

import ai.djl.modality.nlp.generate.CausalLMOutput;
import ai.djl.modality.nlp.generate.GPTConfig;
import ai.djl.modality.nlp.generate.LMBlock;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.pytorch.jni.IValue;
import ai.djl.pytorch.jni.IValueUtils;

import java.util.Arrays;
import java.util.Collections;
import java.util.stream.Collectors;

public class GPT2PtLMBlock extends LMBlock {
    Block[] blocks;
    GPTConfig config;

    public GPT2PtLMBlock(GPTConfig gptConfig, Block[] blocks) {
        config = gptConfig;
        this.blocks = blocks;
    }

    private NDList dummyPastKeyValues(NDArray inputIds, NDManager manager) {
        long numBatch = inputIds.getShape().get(0);
        long kvDim = config.getKvDim();
        int numAttentionHeads = config.getNumAttentionHeads();
        int numLayers = config.getNumLayers();

        NDArray keyOrValue = manager.zeros(new Shape(numBatch, numAttentionHeads, 1, kvDim));
        NDList output = new NDList();
        output.addAll(Collections.nCopies(2 * numLayers, keyOrValue));
        return output;
    }

    /** {@inheritDoc} */
    @Override
    public CausalLMOutput forward(NDList input, NDList pastKeyValues, NDManager manager) {
        // inputIds, positionIds, attentionMask
        long batchSize = input.get(0).getShape().get(0);
        boolean flagDummyKvCach = pastKeyValues == null;
        if (flagDummyKvCach) {
            pastKeyValues = dummyPastKeyValues(input.get(0), manager);
            NDArray attentionMask = input.get(2);
            attentionMask =
                    manager.zeros(new Shape(batchSize, 1), DataType.INT64)
                            .concat(attentionMask, -1);
            input = new NDList(input.get(0), input.get(1), attentionMask);
        }

        IValue[] inputNative =
                input.stream()
                        .map(object -> IValue.from((PtNDArray) object))
                        .toArray(IValue[]::new);
        IValue resultIValue =
                ((PtSymbolBlock) blocks[0])
                        .forward(
                                inputNative[0],
                                inputNative[1],
                                inputNative[2],
                                IValueUtils.toTupleIValue(
                                        pastKeyValues, new long[] {config.getNumLayers(), 2}));

        NDList output = resultIValue.toNDList(manager);
        Arrays.stream(inputNative).forEach(IValue::close);

        NDArray logitsOutput = output.get(0);
        NDList pastKeyValuesOutput = output.subList(1, config.getNumLayers() * 2 + 1);
        NDList hiddenStatesOutput = output.subNDList(config.getNumLayers() * 2 + 2);

        if (flagDummyKvCach) {
            NDIndex index2 = new NDIndex(":, :, 1:, ...");
            pastKeyValuesOutput =
                    new NDList(
                            pastKeyValuesOutput.stream()
                                    .map(object -> object.get(index2))
                                    .collect(Collectors.toList()));
        }

        return new CausalLMOutput(logitsOutput,pastKeyValuesOutput, hiddenStatesOutput);
    }
}
