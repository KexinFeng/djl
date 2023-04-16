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
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.io.IOException;
import java.util.function.Function;
import java.util.stream.Collectors;

public class LMSearch {

    private LMAdapter lmAdapter;

    public NDList contrastiveSearch(
            NDManager manager,
            NDArray inputIds,
            NDArray attentionMask,
            int k,
            float alpha,
            int maxSeqLength) {
        // inputIds: [batchSize, seqLength: t_init]
        // attentionMask: [batchSize, pastSeq]. seq-dim-size = |past_seq| + |inputIds|.
        NDList result = new NDList((int) inputIds.getShape().get(0));
//        result.add(manager.create(new int[]{1, 2}, new Shape(2)));
//        result.set(1, manager.create(new int[]{1}));
        NDArray activeBatchIdx = manager.arange(inputIds.getShape().get(0)).reshape(-1, 1);

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
            assert candidateInputIds.getDataType() == DataType.INT64
                    : "inputIds datatype should be int64";
            assert candidateInputIds.getShape().getShape().length == 2 : "shape not right";

            // [batch, heads, seq_past, feature] -> [batch * topK, head, seq_past, feature]
            NDList kCopyPastKeyValues =
                    (NDList)
                            searchState.pastKeyValues.stream()
                                    .map(ndarray -> ndarray.repeat(0, k))
                                    .collect(Collectors.toList());
            assert kCopyPastKeyValues.get(0).getDataType() == DataType.FLOAT32
                    : "inputIds datatype should be Float32";

            // [batch, seq_past] -> [batch * topK, seq_past] -> [batch * topK, seq_past + 1]
            long numBatch = topKIds.getShape().get(0);
            NDArray kCopyPastAttentionMask = searchState.pastAttentionMask.repeat(0, k);
            kCopyPastAttentionMask =
                    kCopyPastAttentionMask.concat(
                            manager.ones(new Shape(numBatch * k, 1), DataType.INT64), 1);
            assert kCopyPastKeyValues.get(0).getShape().size(-2)
                    == kCopyPastAttentionMask.getShape().size(-1) + 1
                    : "attentionMask_seq = past_seq + new_input_seq";

            // Forward with candidate batch input
            NDList candidateModelInput =
                    prepareInput(
                            candidateInputIds, kCopyPastAttentionMask, kCopyPastKeyValues, manager);
            CausalLMOutput candidateOutput =
                    lmAdapter.forward(candidateModelInput, kCopyPastKeyValues, manager);

            NDArray outputIds =
                    StepGen.ConstrastStepGen(
                            topKIds,
                            searchState.pastAttentionMask, // replaced with initial attentionMask
                            searchState.logits,
                            searchState.pastHiddenStates,
                            candidateOutput.allHiddenStates.get(-1),
                            alpha);

            // Update searchState
            assert candidateOutput.logits.getShape().get(1) == 1
                    : "dimension check: here, outputLogits corresponds to inputSeq == 1";

            long logitsDim = searchState.logits.getShape().get(1);
            long pastSeqLength = searchState.outputIds.getShape().get(1);
            long numHeads = searchState.pastKeyValues.get(0).getShape().get(1);
            long kvDim = searchState.pastKeyValues.get(0).getShape().get(3);
            long hiddenDim = searchState.pastHiddenStates.getShape().get(2);
            NDIndex selectIndex =
                    new NDIndex(
                            "{}, {}, ...",
                            manager.arange(0, numBatch, 1, DataType.INT64),
                            outputIds.flatten());

            // Select from candidateOutput
            // [batch, k, logitsDim] * [batch,] -> [batch, logitDim]
            NDArray nextLogits =
                    candidateOutput.logits.reshape(numBatch, k, logitsDim).get(selectIndex);

            // Select from candidateOutput
            // [batch * k, heads, seq_past, feature]
            Function<NDArray, NDArray> fn =
                    ndarray ->
                            ndarray.reshape(numBatch, k, numHeads, pastSeqLength + 1, kvDim)
                                    .get(selectIndex);
            NDList nextPastKeyValue =
                    (NDList)
                            candidateOutput.pastKeyValuesList.stream()
                                    .map(fn)
                                    .collect(Collectors.toList());

            // To be concatenated into searchState.pastHiddenStates
            // [batch * k, 1, hiddenDim]
            NDArray newHiddenState = candidateOutput.allHiddenStates.get(-1);
            assert newHiddenState.getManager() == manager : "possible leaky memory";
            NDArray nextPastHiddenStates =
                    searchState.pastHiddenStates.concat(
                            newHiddenState.reshape(numBatch, k, 1, hiddenDim).get(selectIndex), 1);

            // To be concatenated into searchState.outputIds
            // [batch, seq_past]
            NDArray nextOutputIds = searchState.outputIds.concat(outputIds, 1);

            NDArray nextPastAttentionMask =
                    searchState.pastAttentionMask.concat(
                            manager.ones(new Shape(numBatch, 1), DataType.INT64), 1);

            searchState =
                    new SearchState(
                            nextLogits,
                            nextPastKeyValue,
                            nextPastHiddenStates,
                            nextOutputIds,
                            nextPastAttentionMask); // can be spared.

            // <EOS>, delete the sentence and add it to result.
            if (searchState.outputIds.getShape().get(1) >= maxSeqLength) {
                break;
            }
        }

//        searchState.outputIds;
        return result;
    }

    private NDList prepareInput(
            NDArray inputIds, NDArray attentionMask, NDList pastKeyValues, NDManager manager) {
        long pastSeqLen = pastKeyValues == null ? 0 : pastKeyValues.get(0).getShape().size(-2);
        NDArray positionIds =
                manager.arange(
                                pastSeqLen,
                                pastSeqLen + inputIds.getShape().get(-1),
                                1,
                                DataType.INT64)
                        .reshape(1, -1)
                        .repeat(0, inputIds.getShape().get(0));

        return new NDList(inputIds, positionIds, attentionMask);
    }
}

class SearchState {
    public int pastSeqLen;

    // [batch, cls]. Only the last logits, used to recall candidate token
    public NDArray logits;

    // [batch, seq_past, hiddenDim]
    // The embed vector of the past seq. seq-dim-size = |past_seq|. Will grow.
    public NDArray pastHiddenStates;

    // [batch, seq_past]
    // The cache of past attentionMask. seq-dim-size == |past_seq| + |inputIds|. Will grow.
    public NDArray pastAttentionMask; // can be spared

    // (k, v) * numLayer,
    // k or v: [batch, heads, seq_past, kvfeature]
    // The cache of past sequence. seq-dim-size == |past_seq| + |inputIds|. Will grow.
    public NDList pastKeyValues;

    // [batch, seq_past]. seq-dim-size == |past_seq| + |inputIds|. Will grow.
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

final class StepGen {
    private StepGen() {}

    public static NDArray ConstrastStepGen(
            NDArray topKIds,
            NDArray attentionMask, // can use initial attentionMask
            NDArray logits,
            NDArray contextHiddenStates,
            NDArray topkHiddenStates,
            float alpha) {
        //  topKIds: [batch, topK]
        //  attentionMask: [batch, past_seq]
        //  logits:  [batch, cls]
        //  contextHiddenStates: [batch, past_seq, dim]
        //  topkHiddenStates: [batch*topK, seq=1, dim]

        long batch = topKIds.getShape().get(0);
        long topK = topKIds.getShape().get(1);
        long hiddenDim = topkHiddenStates.getShape().get(-1);

        // [batch*topK, seq=1, dim] -> [batch, topK, dim]
        topkHiddenStates = topkHiddenStates.reshape(batch, topK, hiddenDim);

        // TODO: add support of Einstein summation:
        // a = torch.randn(batch, past_seq, dim)
        // b = torch.randn(batch, topK, dim)
        // result = torch.einsum('bik,bjk->bij', a, b)

        //  [batch, topK, dim] * [batch, past_seq, dim] -> [batch, topK, past_seq]
        topkHiddenStates.normalize(2, 2);
        contextHiddenStates.normalize(2, 2);
        NDArray cosSimilarity =
                topkHiddenStates.batchMatMul(contextHiddenStates.transpose(0, 2, 1));

        // Deactivate entries (batch_idx, :, zero_attention_idx_slice)

        // [batch, topK, past_seq] -> [batch, topK]
        NDArray topkScorePart1 = cosSimilarity.max(new int[] {2});
        assert topkScorePart1.getShape().getShape().length == 2 : "Wrong output size";
        // [batch, logitDim].gather([batch, topK) -> [batch, topK]
        NDArray topkScorePart2 = logits.gather(topKIds, 1);
        NDArray topkScore = topkScorePart1.mul(alpha).add(topkScorePart2.mul(1-alpha));

        // [batch, topK] => [batch, 1]
        NDArray outputIds = topkScore.argMax(1).expandDims(1);
        assert outputIds.getShape().getShape().length == 2 : "Wrong output size";

        return outputIds;
    }

    public static NDArray GreedyStepGen(NDArray logits) {
        return null;
    }
}
