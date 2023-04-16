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
package ai.djl.translate;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.util.stream.Collectors;

public class LMSearch {

    private LMAdapter lmAdapter;

    public CausalLMOutput contrastiveSearch(
            NDManager manager, NDArray inputIds, NDArray attentionMask, int k) {
        // inputIds: [batchSize, seqLength: t_init]
        // attentionMask: [batchSize, pastSeq]. seq-dim-size = |past_seq| + |inputIds|.
        SearchState searchState = new SearchState();
        while (true) {
            if (searchState.pastKeyValues == null) {
                NDList modelInput = prepareInput(inputIds, attentionMask, null, manager);
                CausalLMOutput output = lmAdapter.forward(modelInput, null, manager);
                NDArray lastLogits = output.logits.get(":, -1, :");
                searchState =
                        new SearchState(
                                lastLogits,
                                output.pastKeyValuesList,
                                output.allHiddenStates.get(-1),
                                inputIds,
                                attentionMask);
            }

            /* Contrastive search loop main part */
            // (1) candidate tokens recall;
            // (2) candidate re-rank by degeneration penalty

            NDArray topKIds = searchState.logits.topK(k, -1, true, false).get(1); // [batch, topK]

            // Generate model inputs and put candidates together into batch

            // [batch, topK] -> [batch * [topK]] -> [[batch * [topK]], seqLength=1]
            NDArray candidateInputIds = topKIds.flatten().reshape(-1, 1);
            assert candidateInputIds.getDataType() == DataType.INT64 : "inputIds datatype should be int64";

            // [batch, heads, seq_past, feature] -> [batch * topK, head, seq_past, feature]
            NDList kCopyPastKeyValues =
                    (NDList)
                            searchState.pastKeyValues.stream()
                                    .map(ndarray -> ndarray.repeat(0, k))
                                    .collect(Collectors.toList());
            assert kCopyPastKeyValues.get(0).getDataType() == DataType.FLOAT32 : "inputIds datatype should be Float32";

            // [batch, seq_past] -> [batch * topK, seq_past] -> [batch * topK, seq_past + 1]
            long numBatch = searchState.logits.getShape().get(0);
            NDArray kCopyPastAttentionMask = searchState.pastAttentionMask.repeat(0, k);
            kCopyPastAttentionMask =
                    kCopyPastAttentionMask.concat(
                            manager.ones(new Shape(numBatch * k, 1), DataType.INT64), 1);

            assert kCopyPastKeyValues.get(0).getShape().size(-2)
                            == kCopyPastAttentionMask.getShape().size(-1) + 1
                    : "attentionMask_seq = past_seq + new_input_seq";

            NDList candidateModelInput =
                    prepareInput(
                            candidateInputIds, kCopyPastAttentionMask, kCopyPastKeyValues, manager);
            CausalLMOutput candidateModelOutput =
                    lmAdapter.forward(candidateModelInput, kCopyPastKeyValues, manager);



            if (true) {
                break;
            }
        }

        return null;
    }

    private NDList prepareInput(
            NDArray inputIds, NDArray attentionMask, NDList pastKeyValues, NDManager manager) {
        long numBatch = inputIds.getShape().get(0);
        long pastSeqLen = pastKeyValues == null ? 0 : pastKeyValues.get(0).getShape().size(-2);
        NDArray positionIds =
                manager.arange(
                                pastSeqLen,
                                pastSeqLen + inputIds.getShape().get(-1),
                                1,
                                DataType.INT64)
                        .reshape(1, -1)
                        .repeat(0, numBatch);

        return new NDList(inputIds, positionIds, attentionMask);
    }
}

class SearchState {
    // [batch, cls]. Only the last logits, used to recall candidate token
    public NDArray logits;

    // [batch, seq, dim]
    // The vec. rep. of the past seq. seq-dim-size = |past_seq|. Will grow.
    public NDArray pastHiddenStates;

    // (k, v) * numLayer, k or v: [batch, heads, seq_past, feature]
    // The cache of past sequence. seq-dim-size == |past_seq| + |inputIds|. Will grow.
    public NDList pastKeyValues;

    // [batch, seq_past]
    // The cache of past attentionMask. seq-dim-size == |past_seq| + |inputIds|. Will grow.
    public NDArray pastAttentionMask;

    // [batch, seq]. past_seq
    public NDArray outputIds;

    SearchState() {}
    SearchState(
            NDArray logits,
            NDList pastKeyValues,
            NDArray pastHiddenStates,
            NDArray outputIds,
            NDArray pastAttentionMask) {
        this.logits = logits;
        this.pastKeyValues = pastKeyValues;
        this.pastHiddenStates = pastHiddenStates;
        this.outputIds = outputIds;
        this.pastAttentionMask = pastAttentionMask;
    }
}
