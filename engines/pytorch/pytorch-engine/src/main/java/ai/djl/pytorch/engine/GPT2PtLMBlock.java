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
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Block;
import ai.djl.pytorch.jni.IValue;
import ai.djl.pytorch.jni.IValueUtils;

import java.util.Arrays;

public class GPT2PtLMBlock extends LMBlock {
    Block[] blocks;
    GPTConfig config;

    public GPT2PtLMBlock(GPTConfig gptConfig, Block[] blocks) {
        config = gptConfig;
        this.blocks = blocks;
    }

    /** {@inheritDoc} */
    @Override
    public CausalLMOutput forward(NDList input, NDList pastKeyValues, NDManager manager) {
        IValue[] inputNative =
                input.stream()
                        .map(object -> IValue.from((PtNDArray) object))
                        .toArray(IValue[]::new);

        IValue resultIValue;
        if (pastKeyValues == null) {
            resultIValue = ((PtSymbolBlock) blocks[0]).forward(inputNative);
        } else {
            resultIValue =
                    ((PtSymbolBlock) blocks[1])
                            .forward(
                                    inputNative[0],
                                    inputNative[1],
                                    inputNative[2],
                                    IValueUtils.toTupleIValue(
                                            pastKeyValues, new long[] {config.numLayers, 2}));
        }
        NDList output = resultIValue.toNDList(manager);
        Arrays.stream(inputNative).forEach(IValue::close);

        return new CausalLMOutput(
                output.get(0),
                output.subList(1, config.numLayers * 2 + 1),
                output.subNDList(config.numLayers * 2 + 2));
    }
}
