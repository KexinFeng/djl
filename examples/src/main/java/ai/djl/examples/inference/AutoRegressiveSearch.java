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
import ai.djl.Model;
import ai.djl.modality.nlp.generate.LMBlock;
import ai.djl.modality.nlp.generate.LMSearch;
import ai.djl.modality.nlp.generate.SearchConfig;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Block;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.util.Pair;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;

public final class AutoRegressiveSearch {

    LMBlock lmBlockPt;

    LMBlock lmBlockOnnx;

    List<Model> modelsPt;

    List<Model> modelsOnnx;

    private static final Logger logger = LoggerFactory.getLogger(AutoRegressiveSearch.class);

    public AutoRegressiveSearch()
            throws ModelNotFoundException, MalformedModelException, IOException {
        String[] modelUrls = {
            "https://djl-misc.s3.amazonaws.com/test/models/gpt2/gpt2_init.pt.zip",
            "https://djl-misc.s3.amazonaws.com/test/models/gpt2/gpt2.pt.zip"
        };
        Pair<Block, List<Model>> result = LLMBlock.getLMBlock(modelUrls, "PyTorch", "GPT2");
        lmBlockPt = (LMBlock) result.getKey();
        modelsPt = result.getValue();

        modelUrls =
                new String[] {"https://djl-misc.s3.amazonaws.com/test/models/gpt2/gpt2.onnx.zip"};
        result = LLMBlock.getLMBlock(modelUrls, "OnnxRuntime", "GPT2");
        lmBlockOnnx = (LMBlock) result.getKey();
        modelsOnnx = result.getValue();
    }

    public void main(String[] args)
            throws ModelNotFoundException, MalformedModelException, IOException {
        mainContrastivePt(args);
        mainGreedyPt(args);
        mainBeamPt(args);
        mainBeamOnnx(args);
    }

    public boolean mainContrastivePt(String[] args)
            throws ModelNotFoundException, MalformedModelException, IOException {
        LMBlock lmBlock = lmBlockPt;
        try (NDManager manager = NDManager.newBaseManager()) {
            SearchConfig config = new SearchConfig();
            config.maxSeqLength = 60;
            config.alpha = 0.6f;
            config.k = 3;

            // [r'DeepMind Company is',
            // r'Memories follow me left and right. I can']
            NDArray inputIds =
                    manager.create(
                            new long[][] {
                                {220, 220, 220, 220, 220, 220, 29744, 28478, 5834, 318},
                                {13579, 1749, 1061, 502, 1364, 290, 826, 13, 314, 460}
                            });
            config.padTokenId = 220;
            config.suffixPadding = false;

            LMSearch lmSearch;
            lmSearch = new LMSearch(lmBlock, "constrastive", config);

            NDArray output = lmSearch.contrastiveSearch(inputIds);
            NDArray expected =
                    manager.create(
                            new long[][] {
                                {284, 8494, 3716, 2761, 11, 884, 355, 1692, 1535, 11},
                                {4436, 329, 257, 2910, 1332, 13, 632, 373, 257, 3487}
                            });
            return output.get(":, -10:").equals(expected);
        }
    }

    public boolean mainGreedyPt(String[] args)
            throws ModelNotFoundException, MalformedModelException, IOException {
        LMBlock lmBlock = lmBlockPt;
        try (NDManager manager = NDManager.newBaseManager()) {

            SearchConfig config = new SearchConfig();
            config.maxSeqLength = 60;

            // [r'DeepMind Company is',
            // r'Memories follow me left and right. I can']
            NDArray inputIds =
                    manager.create(
                            new long[][] {
                                {220, 220, 220, 220, 220, 220, 29744, 28478, 5834, 318},
                                {13579, 1749, 1061, 502, 1364, 290, 826, 13, 314, 460}
                            });
            config.padTokenId = 220;
            config.suffixPadding = false;

            LMSearch lmSearch = new LMSearch(lmBlock, "greedy", config);

            NDArray output = lmSearch.greedySearch(inputIds);
            NDArray expected =
                    manager.create(
                            new long[][] {
                                {389, 635, 257, 3756, 10131, 286, 6190, 9552, 8136, 329},
                                {257, 6576, 13, 314, 460, 470, 3505, 262, 938, 640}
                            });
            return output.get(":, -10:").equals(expected);
        }
    }

    public boolean mainBeamPt(String[] args)
            throws ModelNotFoundException, MalformedModelException, IOException {
        LMBlock lmBlock = lmBlockPt;
        try (NDManager manager = NDManager.newBaseManager()) {

            SearchConfig config = new SearchConfig();
            config.maxSeqLength = 60;
            config.beam = 3;

            // [r'DeepMind Company is',
            // r'Memories follow me left and right. I can']
            NDArray inputIds =
                    manager.create(
                            new long[][] {
                                {50256, 50256, 50256, 50256, 50256, 50256, 29744, 28478, 5834, 318},
                                {13579, 1749, 1061, 502, 1364, 290, 826, 13, 314, 460}
                            });
            config.padTokenId = 50256;
            config.suffixPadding = false;

            LMSearch lmSearch = new LMSearch(lmBlock, "beam", config);

            NDArray output = lmSearch.beamSearch(inputIds);
            NDArray expected =
                    manager.create(
                            new long[] {2267, 290, 2478, 13, 198, 198, 5122, 4365, 318, 284});
            return output.get("0, -10:").equals(expected);
        }
    }

    public boolean mainBeamOnnx(String[] args)
            throws ModelNotFoundException, MalformedModelException, IOException {
        LMBlock lmBlock = lmBlockOnnx;
        try (NDManager manager = NDManager.newBaseManager()) {

            SearchConfig config = new SearchConfig();
            config.maxSeqLength = 60;
            config.beam = 3;

            // [r'DeepMind Company is',
            // r'Memories follow me left and right. I can']
            NDArray inputIds =
                    manager.create(
                            new long[][] {
                                {220, 220, 220, 220, 220, 220, 29744, 28478, 5834, 318},
                                {13579, 1749, 1061, 502, 1364, 290, 826, 13, 314, 460}
                                //                                {220, 29744, 28478, 5834, 318}
                            });
            config.padTokenId = 220;
            config.suffixPadding = false;
            // The positionIds is not effective in onnx model traced from huggingface optimum.

            LMSearch lmSearch = new LMSearch(lmBlock, "beam", config);
            NDArray output = lmSearch.beamSearch(inputIds);

            logger.info(
                    "Notice: with OnnxRuntime model, it doesn't take positionId yet (only"
                            + " attentionMask is effective), so the output is pathologic.");

            NDArray expected =
                    manager.create(
                            new long[] {
                                10766, 10766, 10766, 10766, 10766, 10766, 10766, 10766, 10766, 10766
                            });
            return output.get("0, -10:").equals(expected);
        }
    }
}