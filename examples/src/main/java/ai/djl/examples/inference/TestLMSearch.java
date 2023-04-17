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
package ai.djl.examples.inference;

import ai.djl.MalformedModelException;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.translate.GPTConfig;
import ai.djl.translate.LMAdapter;
import ai.djl.translate.LMSearch;
import ai.djl.translate.SearchConfig;

import java.io.IOException;

public final class TestLMSearch {

    private TestLMSearch() {}

    public static void main(String[] args) {
        mainPt(args);
    }

    public static void mainPt(String[] args) {
//        String[] modelUrls = {
//            "/Users/fenkexin/Desktop/tasks/HuggingFaceQa_relavant/transformer/traced_GPT2_init_hidden.pt",
//            "/Users/fenkexin/Desktop/tasks/HuggingFaceQa_relavant/transformer/traced_GPT2_hidden.pt"
//        };
//        GPTConfig gptConfig = new GPTConfig(modelUrls);
//        gptConfig.numAttentionHeads = 20;
//        gptConfig.numLayers = 36;
//        gptConfig.hiddenStateDim = 768;
//        gptConfig.logitsDim = 50257;
//        gptConfig.kvDim = 64;

        String[] modelUrls = {
                "/Users/fenkexin/Desktop/tasks/HuggingFaceQa_relavant/transformer/models/traced_GPT2_init_hidden.pt",
                "/Users/fenkexin/Desktop/tasks/HuggingFaceQa_relavant/transformer/models/traced_GPT2_hidden.pt"
        };
        GPTConfig gptConfig = new GPTConfig(modelUrls);

        try (LMAdapter lmAdapter = Engine.getEngine("PyTorch").newLMAdapter("GPT2", gptConfig);
                NDManager manager = NDManager.newBaseManager()) {

            LMSearch lmSearch;
            lmSearch = new LMSearch(lmAdapter);
            SearchConfig config = new SearchConfig();

            // prefix_text = [r'DeepMind Company is',
            //                r'Hi there']
            long[][] inputArray =
                    new long[][] {
                        {29744, 28478, 5834, 318},
                        {17250, 612, 50256, 50256}
                    };

            NDArray inputIds = manager.create(inputArray);
            NDArray attentionMask =
                    manager.ones(new Shape(1, inputIds.getShape().get(-1)), DataType.INT64)
                            .reshape(1, -1)
                            .repeat(0, inputArray.length);

            long[][] attentionMaskSlice = new long[inputArray.length][2];
            int initSeqSize = inputArray[0].length;
            for (int i = 0; i < inputArray.length; i++) {
                long[] aSequence = attentionMask.get("{},:", i).toLongArray();
                int idx = 0;
                while (idx < initSeqSize) {
                    if (aSequence[idx] == config.padTokenId) break;
                    idx++;
                }
//                attentionMaskSlice[i][0] = idx;
                attentionMaskSlice[i][0] = initSeqSize;
                attentionMaskSlice[i][1] = initSeqSize;
//                attentionMask.set(new NDIndex("{},{}:{}", i, idx, initSeqSize), 0);
                attentionMask.set(new NDIndex("{},{}:{}", i, initSeqSize, initSeqSize), 0);
            }

            NDArray output =
                    lmSearch.contrastiveSearch(
                            manager, inputIds, attentionMask, attentionMaskSlice, config);
            System.out.println(output);
            int dbstop = 1;

        } catch (ModelNotFoundException | MalformedModelException | IOException e) {
            throw new RuntimeException(e);
        }
    }
}
