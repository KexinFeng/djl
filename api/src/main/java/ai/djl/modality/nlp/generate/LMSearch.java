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
import ai.djl.ndarray.NDScope;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

import java.util.function.Function;
import java.util.stream.Collectors;

public class LMSearch extends AbstractBlock {

    private String searchName;
    private SearchConfig config;
    private LMBlock lmBlock;

    public NDArray positionOffset;

    public LMSearch(LMBlock lmBlock, String searchName, SearchConfig searchConfig) {
        this.lmBlock = lmBlock;
        this.searchName = searchName;
        this.config = searchConfig;
    }

    public NDArray greedySearch(NDArray inputIds) {
        NDArray attentionMask = prepareAttentionMaskOffset(inputIds, config);
        NDManager manager = inputIds.getManager();
        GreedyBatchTensorList searchState =
                new GreedyBatchTensorList(inputIds, null, null, attentionMask);
        while (true) {
            try (NDScope scope = new NDScope()) {
                scope.suppressNotUsedWarning();

                long pastSeqLength =
                        searchState.pastOutputIds == null
                                ? 0
                                : searchState.pastOutputIds.getShape().get(-1);
                NDList modelInput =
                        prepareInput(
                                searchState.nextInputIds,
                                searchState.pastAttentionMask,
                                pastSeqLength,
                                1);
                CausalLMOutput modelOutput =
                        lmBlock.forward(modelInput, searchState.pastKeyValues, manager);

                NDArray outputIds = StepGeneration.greedyStepGen(modelOutput.logits);

                // Update searchState
                if (searchState.pastOutputIds == null) {
                    searchState.pastOutputIds = searchState.nextInputIds;
                } else {
                    searchState.pastOutputIds =
                            searchState.pastOutputIds.concat(searchState.nextInputIds, 1);
                }
                searchState.nextInputIds = outputIds;
                searchState.pastKeyValues = modelOutput.pastKeyValuesList;
                searchState.pastAttentionMask =
                        searchState.pastAttentionMask.concat(
                                manager.ones(
                                        new Shape(inputIds.getShape().get(0), 1), DataType.INT64),
                                1);

                // memory management
                NDScope.unregister(
                        searchState.nextInputIds,
                        searchState.pastAttentionMask,
                        searchState.pastOutputIds);
                NDScope.unregister(searchState.pastKeyValues);
            }

            // Termination Criteria
            // TODO: <EOS>, delete the sentence and add it to result.
            if (searchState.pastOutputIds.getShape().get(1) + 1 >= config.maxSeqLength) {
                break;
            }
        }
        return searchState.pastOutputIds.concat(searchState.nextInputIds, 1);
    }

    // https://huggingface.co/blog/how-to-generate
    public NDArray beamSearch(NDArray inputIds) {
        NDArray attentionMask = prepareAttentionMaskOffset(inputIds, config);
        NDManager manager = inputIds.getManager();
        long numBeam = config.beam;
        long numBatch = inputIds.getShape().get(0);
        BeamBatchTensorList searchState = new BeamBatchTensorList();

        long numHeads = 0;
        long kvDim = 0;
        while (true) {
            if (searchState.pastAttentionMask == null) {
                // Initial beams
                NDList modelInput = prepareInput(inputIds, attentionMask, 0, 1);
                CausalLMOutput modelOutput = lmBlock.forward(modelInput, null, manager);

                // [batch, probDim]
                NDArray allProbs = modelOutput.logits.get(":, -1, :").softmax(1);

                // [batch, beam]
                NDList topK = allProbs.topK((int) numBeam, -1, true, false);
                NDArray outputIds = topK.get(1).expandDims(2);
                NDArray lastProbs = topK.get(0).normalize(1, 1);
                assert outputIds.getShape().getShape().length == 3 : "Wrong shape";
                assert lastProbs.getShape().getShape().length == 2 : "Wrong Shape";

                // [batch, beam, seq + 1]
                attentionMask =
                        attentionMask
                                .concat(manager.ones(new Shape(numBatch, 1), DataType.INT64), -1)
                                .expandDims(1)
                                .repeat(1, numBeam);

                // [batch, beam, heads, seq_past, kvFeature]
                Function<NDArray, NDArray> fn = ndarray -> ndarray.expandDims(1).repeat(1, numBeam);
                NDList pastKeyValues =
                        new NDList(
                                modelOutput.pastKeyValuesList.stream()
                                        .map(fn)
                                        .collect(Collectors.toList()));
                // [batch, beam, seq_past]
                NDArray pastOutputIds = inputIds.expandDims(1).repeat(1, numBeam);

                searchState =
                        new BeamBatchTensorList(
                                outputIds, pastOutputIds, pastKeyValues, attentionMask, lastProbs);

                numHeads = pastKeyValues.get(0).getShape().get(-3);
                kvDim = pastKeyValues.get(0).getShape().get(-1);
            }

            try (NDScope scope = new NDScope()) {
                scope.suppressNotUsedWarning();

                long pastSeqLength = searchState.pastOutputIds.getShape().get(-1);
                NDList modelInput =
                        prepareInput(
                                searchState.nextInputIds.reshape(numBatch * numBeam, 1),
                                searchState.pastAttentionMask.reshape(numBatch * numBeam, -1),
                                pastSeqLength,
                                config.beam);

                final long finalNumHeads = numHeads;
                final long finalKvDim = kvDim;
                Function<NDArray, NDArray> fn =
                        ndarray ->
                                ndarray.reshape(
                                        numBatch * numBeam,
                                        finalNumHeads,
                                        pastSeqLength,
                                        finalKvDim);
                NDList pastKeyValues =
                        new NDList(
                                searchState.pastKeyValues.stream()
                                        .map(fn)
                                        .collect(Collectors.toList()));
                CausalLMOutput modelOutput = lmBlock.forward(modelInput, pastKeyValues, manager);

                NDList generatedOutput =
                        StepGeneration.beamStepGeneration(
                                searchState.lastProbs, modelOutput.logits, numBatch, numBeam);

                // Update searchState
                searchState = updateSearchState(searchState, modelOutput, generatedOutput, manager);

                // Memory management
                NDScope.unregister(
                        searchState.nextInputIds,
                        searchState.pastOutputIds,
                        searchState.pastAttentionMask,
                        searchState.lastProbs);
                NDScope.unregister(searchState.pastKeyValues);
            }

            // Termination Criteria
            // TODO: <EOS>, delete the sentence and add it to result.
            if (searchState.pastOutputIds.getShape().get(-1) + 1 >= config.maxSeqLength) {
                break;
            }
        }

        return searchState
                .pastOutputIds
                .concat(searchState.nextInputIds, -1)
                .reshape(numBatch * numBeam, -1);
    }

    // https://huggingface.co/blog/introducing-csearch
    public NDArray contrastiveSearch(NDArray inputIds) {
        // inputIds: [batchSize, seqLength: t_init]
        // attentionMask: [batchSize, pastSeq]. seq-dim-size = |past_seq| + |inputIds|.

        NDManager manager = inputIds.getManager();
        NDArray attentionMask = prepareAttentionMaskOffset(inputIds, config);
        ContrastiveBatchTensorList searchState = new ContrastiveBatchTensorList();
        while (true) {
            if (searchState.pastKeyValues == null) {
                NDList modelInput = prepareInput(inputIds, attentionMask, 0, 1);
                CausalLMOutput output = lmBlock.forward(modelInput, null, manager);
                NDArray lastLogits = output.logits.get(":, -1, :");
                searchState =
                        new ContrastiveBatchTensorList(
                                inputIds,
                                attentionMask,
                                output.allHiddenStates.get(0),
                                lastLogits,
                                output.pastKeyValuesList,
                                new long[] {});
            }

            /* Contrastive search loop main part */
            // (1) candidate tokens recall;
            // (2) candidate re-rank by degeneration penalty

            try (NDScope scope = new NDScope()) {
                scope.suppressNotUsedWarning();

                NDArray topKIds =
                        searchState.logits.topK(config.k, -1, true, false).get(1); // [batch, topK]

                // Generate model inputs and put candidates together into batch
                // [batch, topK] -> [batch * [topK]] -> [[batch * [topK]], seqLength=1]
                NDArray candidateInputIds = topKIds.flatten().reshape(-1, 1);
                assert candidateInputIds.getDataType() == DataType.INT64
                        : "inputIds datatype should be int64";
                assert candidateInputIds.getShape().getShape().length == 2 : "shape not right";

                // [batch, heads, seq_past, feature] -> [batch * topK, head, seq_past, feature]
                NDList kCopyPastKeyValues =
                        new NDList(
                                searchState.pastKeyValues.stream()
                                        .map(ndarray -> ndarray.repeat(0, config.k))
                                        .collect(Collectors.toList()));
                assert kCopyPastKeyValues.get(0).getDataType() == DataType.FLOAT32
                        : "inputIds datatype should be Float32";

                // [batch, seq_past] -> [batch * topK, seq_past] -> [batch * topK, seq_past + 1]
                long numBatch = topKIds.getShape().get(0);
                NDArray kCopyPastAttentionMask = searchState.pastAttentionMask.repeat(0, config.k);
                kCopyPastAttentionMask =
                        kCopyPastAttentionMask.concat(
                                manager.ones(new Shape(numBatch * config.k, 1), DataType.INT64), 1);
                assert kCopyPastKeyValues.get(0).getShape().get(-2) + 1
                                == kCopyPastAttentionMask.getShape().get(-1)
                        : "attentionMask_seq = past_seq + new_input_seq";

                // Forward with candidates in batch input
                NDList candidateModelInput =
                        prepareInput(
                                candidateInputIds,
                                kCopyPastAttentionMask,
                                searchState.pastOutputIds.getShape().get(-1),
                                config.k);
                CausalLMOutput candidateOutput =
                        lmBlock.forward(candidateModelInput, kCopyPastKeyValues, manager);

                NDList generatedOutput =
                        StepGeneration.constrastiveStepGeneration(
                                topKIds,
                                searchState.logits,
                                searchState.pastHiddenStates,
                                candidateOutput.allHiddenStates.get(0),
                                positionOffset,
                                config.alpha);

                // Update searchState for next loop
                searchState =
                        updateSearchState(searchState, candidateOutput, generatedOutput, manager);

                // Memory
                NDScope.unregister(
                        searchState.pastOutputIds,
                        searchState.pastAttentionMask,
                        searchState.logits,
                        searchState.pastHiddenStates);
                NDScope.unregister(searchState.pastKeyValues);
            }

            // TODO: <EOS>, delete the sentence and add it to result.
            if (searchState.pastOutputIds.getShape().get(1) >= config.maxSeqLength) {
                break;
            }
        }

        return searchState.pastOutputIds;
    }

    private static BeamBatchTensorList updateSearchState(
            BeamBatchTensorList searchState,
            CausalLMOutput modelOutput,
            NDList generatedOutput,
            NDManager manager) {

        NDList pastKeyValues = searchState.pastKeyValues;
        long numHeads = pastKeyValues.get(0).getShape().get(-3);
        long kvDim = pastKeyValues.get(0).getShape().get(-1);
        long numBatch = searchState.pastOutputIds.getShape().get(0);
        long numBeam = searchState.pastOutputIds.getShape().get(1);
        long pastSeqLength = searchState.pastOutputIds.getShape().get(-1);

        NDArray nextInputIds = generatedOutput.get(0);
        assert nextInputIds.getShape().getShape().length == 3 : "Wrong Shape";
        NDArray newProbs = generatedOutput.get(1);
        // [batch, beamNew]
        NDArray sourceBeamSelected = generatedOutput.get(2);
        // Act on [batch, beam, ...] dimension and the output will be [batch, beam, ...]
        NDIndex sourceBeamIndex =
                new NDIndex(
                        "{}, {}, ...",
                        manager.arange(0, numBatch, 1, DataType.INT64)
                                .expandDims(1)
                                .repeat(1, numBeam),
                        sourceBeamSelected);

        // A simple concatenation is not enough. During the beam selection process, some source
        // beams are selected several times while some source beams are not selected even once.
        // The pastOutput should be reselected to have the right correspondence to the
        // newInputIds.
        NDArray pastOutputIds =
                searchState.pastOutputIds.concat(searchState.nextInputIds, -1).get(sourceBeamIndex);
        Function<NDArray, NDArray> fn =
                ndarray ->
                        ndarray.reshape(numBatch, numBeam, numHeads, pastSeqLength + 1, kvDim)
                                .get(sourceBeamIndex);
        pastKeyValues =
                new NDList(
                        modelOutput.pastKeyValuesList.stream()
                                .map(fn)
                                .collect(Collectors.toList()));

        NDArray pastAttentionMask =
                searchState
                        .pastAttentionMask
                        .concat(manager.ones(new Shape(numBatch, numBeam, 1), DataType.INT64), -1)
                        .get(sourceBeamIndex);

        return new BeamBatchTensorList(
                nextInputIds, pastOutputIds, pastKeyValues, pastAttentionMask, newProbs);
    }

    private static ContrastiveBatchTensorList updateSearchState(
            ContrastiveBatchTensorList searchState,
            CausalLMOutput candidateOutput,
            NDList generatedOutput,
            NDManager manager) {
        // Update searchState for next iteration
        assert candidateOutput.logits.getShape().get(1) == 1
                : "dimension check: here, outputLogits corresponds to inputSeq == 1";
        long numBatch = searchState.logits.getShape().get(0);
        long logitsDim = searchState.logits.getShape().get(1);
        long pastSeqLengthPriorUpdate = searchState.pastOutputIds.getShape().get(1);
        long numHeads = searchState.pastKeyValues.get(0).getShape().get(1);
        long kvDim = searchState.pastKeyValues.get(0).getShape().get(3);
        long hiddenDim = searchState.pastHiddenStates.getShape().get(2);
        long k = candidateOutput.logits.getShape().get(0) / numBatch;

        // [batch, 1]
        NDArray select = generatedOutput.get(1);
        NDIndex selectIndex =
                new NDIndex(
                        "{}, {}, ...",
                        manager.arange(0, numBatch, 1, DataType.INT64),
                        select.flatten());

        // Take from candidateOutput
        // [batch, k, inputSeq=1, logitsDim] --select--> [batch, logitDim]
        NDArray nextLogits =
                candidateOutput.logits.reshape(numBatch, k, logitsDim).get(selectIndex);

        // Take from candidateOutput
        // [batch * k, heads, seq_past, feature] --select--> [batch, heads, seq_past, feature]
        Function<NDArray, NDArray> fn =
                ndarray ->
                        ndarray.reshape(numBatch, k, numHeads, pastSeqLengthPriorUpdate + 1, kvDim)
                                .get(selectIndex);
        NDList nextPastKeyValue =
                new NDList(
                        candidateOutput.pastKeyValuesList.stream()
                                .map(fn)
                                .collect(Collectors.toList()));

        // To be concatenated into searchState.pastHiddenStates
        // [batch * k, inputSeq=1, hiddenDim]
        NDArray newHiddenState = candidateOutput.allHiddenStates.get(0);
        assert newHiddenState.getManager() == manager : "possible leaky memory";
        NDArray nextPastHiddenStates =
                searchState.pastHiddenStates.concat(
                        newHiddenState.reshape(numBatch, k, 1, hiddenDim).get(selectIndex), 1);

        // To be concatenated into searchState.outputIds
        // [batch, seq_past]
        NDArray outputIds = generatedOutput.get(0);
        NDArray nextOutputIds = searchState.pastOutputIds.concat(outputIds, 1);

        // [batch, seq_past]
        NDArray nextPastAttentionMask =
                searchState.pastAttentionMask.concat(
                        manager.ones(new Shape(numBatch, 1), DataType.INT64), 1);

        return new ContrastiveBatchTensorList(
                nextOutputIds,
                nextPastAttentionMask,
                nextPastHiddenStates,
                nextLogits,
                nextPastKeyValue,
                new long[] {});
    }

    private NDArray prepareAttentionMaskOffset(NDArray inputIds, SearchConfig config) {
        // prepare attentionMask and positionOffset
        // Used to initialize the search
        boolean suffixPadding = config.suffixPadding;
        NDManager manager = inputIds.getManager();
        int numBatch = (int) inputIds.getShape().get(0);
        int initSeqSize = (int) inputIds.getShape().get(1);
        NDArray attentionMask =
                manager.ones(new Shape(1, inputIds.getShape().get(-1)), DataType.INT64)
                        .reshape(1, -1)
                        .repeat(0, numBatch);

        // Linear search from left to find the first position that's not padTokenId.
        long[][] offset = new long[numBatch][1];
        for (int i = 0; i < numBatch; i++) {
            long[] aSequence = inputIds.get("{},:", i).toLongArray();
            int idx = 0;
            while (idx < initSeqSize) {
                if (suffixPadding && aSequence[idx] == config.padTokenId
                        || !suffixPadding && aSequence[idx] != config.padTokenId) {
                    break;
                }
                idx++;
            }
            attentionMask.set(
                    new NDIndex(
                            "{},{}:{}",
                            i,
                            suffixPadding ? idx : 0,
                            suffixPadding ? initSeqSize : idx),
                    0);
            if (!suffixPadding) {
                offset[i][0] = idx;
            }
        }
        positionOffset = manager.create(offset);
        return attentionMask;
    }

    private NDList prepareInput(
            NDArray inputIds, NDArray attentionMask, long pastSeqLength, int repeat) {
        // Pack the model input
        NDArray positionIds =
                inputIds.getManager()
                        .arange(
                                pastSeqLength,
                                pastSeqLength + inputIds.getShape().get(-1),
                                1,
                                DataType.INT64)
                        .expandDims(0)
                        .repeat(0, inputIds.getShape().get(0));

        NDArray positionIdsShifted = positionIds.subi(positionOffset.repeat(0, repeat));
        positionIds = positionIdsShifted.maximum(positionIdsShifted.zerosLike());

        return new NDList(inputIds, positionIds, attentionMask);
    }

    public NDArray forward(NDArray inputIds) {
        switch (searchName) {
            case "greedy":
                return greedySearch(inputIds);
            case "beam":
                return beamSearch(inputIds);
            case "contrastive":
                return contrastiveSearch(inputIds);
            default:
                throw new IllegalArgumentException(
                        "searchName not correctly specified. Please choose among: {greedy, beam,"
                                + " contrastive}");
        }
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        return new NDList(forward(inputs.get(0)));
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[] {};
    }
}
