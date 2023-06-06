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
package ai.djl.modality.nlp.generate;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * This is a wrapper over the model files from different sources, e.g. gpt2.pt, gpt2.onnx, etc. This
 * interface is an abstraction of the causal language model, which in essence is a conditional
 * probability function: p_\theta(v_t | x_{< t})}, v_t \in V, i.e. given the past tokens up to a
 * certain time x_{< t}, the probability that the next token is v, taken from a vocabulary set V.
 * \theta is the model's weight. This function can take an input sequence `inputIds`, whose length
 * can be greater than one. In this case, the output is still p_\theta(v_i | x_{< i})}, i in
 * range(|inputIds|). This means for each i, the output probability is conditional on the past
 * sequence up to i.
 */
public abstract class LMBlock extends AbstractBlock {

    /**
     * @param input input
     * @param pastKeyValues past_key_values
     * @param manager manager
     * @return CausalLMOutput
     */
    public abstract CausalLMOutput forward(NDList input, NDList pastKeyValues, NDManager manager);

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        // inputIds, positionIds, attentionMask
        CausalLMOutput output =
                forward(inputs.subList(0, 3), inputs.subNDList(3), inputs.getManager());
        return new NDList(output.getLogits())
                .addAll(output.getAllHiddenStates()) // allHiddenStates could be null
                .addAll(output.getPastKeyValuesList());
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray inputIds = manager.ones(inputShapes[0], DataType.INT64);
            NDArray positionIds =
                    manager.arange(0, inputIds.getShape().size(-1), 1, DataType.INT64)
                            .reshape(1, -1)
                            .repeat(0, inputIds.getShape().get(0));
            NDArray attentionMask = manager.ones(positionIds.getShape(), DataType.INT64);
            NDList input = new NDList(inputIds, positionIds, attentionMask);

            NDList result = forwardInternal(new ParameterStore(manager, false), input, false, null);
            return result.stream().map(NDArray::getShape).toArray(Shape[]::new);
        }
    }
}
