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
import ai.djl.engine.Engine;
import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.generate.GPTConfig;
import ai.djl.modality.nlp.generate.LMBlock;
import ai.djl.modality.nlp.generate.LMSearch;
import ai.djl.modality.nlp.generate.SearchConfig;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

public final class GPTInference {

    private static final Logger logger = LoggerFactory.getLogger(ImageClassification.class);

    private GPTInference() {}

    public static void main(String[] args)
            throws ModelNotFoundException, MalformedModelException, IOException,
                    TranslateException {
        //        testOnnx();
        testPtGreedy();
    }

    private static void testPtGreedy()
            throws ModelNotFoundException, MalformedModelException, IOException {
        String[] modelUrls = {
            "https://djl-misc.s3.amazonaws.com/test/models/gpt2/gpt2_init.pt.zip",
            "https://djl-misc.s3.amazonaws.com/test/models/gpt2/gpt2.pt.zip"
        };
        Block[] blocks = new Block[modelUrls.length];
        List<Model> models = new LinkedList<>();
        for (int i = 0; i < modelUrls.length; i++) {
            Criteria<NDList, NDList> criteria =
                    Criteria.builder()
                            .setTypes(NDList.class, NDList.class)
                            .optModelUrls(modelUrls[i])
                            .optEngine("PyTorch")
                            .optProgress(new ProgressBar())
                            .build();
            Model model = criteria.loadModel();
            blocks[i] = model.getBlock();
            models.add(model);
        }
        LMBlock lmBlock = Engine.getEngine("PyTorch").newLMBlock("GPT2", new GPTConfig(), blocks);

        SearchConfig config = new SearchConfig();
        config.maxSeqLength = 60;
        LMSearch lmSearch = new LMSearch(lmBlock, "greedy", config);

        String[] input = {"DeepMind Company is"};
        try (NDManager manager = NDManager.newBaseManager("PyTorch");
                HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance("gpt2")) {
            Encoding encoding = tokenizer.encode(input);
            long[] inputIdsLong = encoding.getIds();
            NDArray inputIds = manager.create(inputIdsLong).expandDims(0);

            NDArray output = lmSearch.forward(inputIds).get(":, -10:");

            String outputString = tokenizer.decode(output.toLongArray());
            String expected = "are also a leading provider of advanced AI solutions for";

            logger.info("{}", expected.equals(outputString));
        }
        // According to the last code review meeting, the conclusion was to only put tokenizer's encoding
        // and decoding part into the translator's pre/post process. That is also why Zach proposed
        // to make LMSearch integrate AbstractBlock so that it will be called by predictor.

        models.forEach(Model::close);
    }

    private static void testOnnx()
            throws ModelNotFoundException, MalformedModelException, IOException,
                    TranslateException {
        String url = "https://djl-misc.s3.amazonaws.com/test/models/gpt2/gpt2.onnx.zip";
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelUrls(url)
                        .optEngine("OnnxRuntime")
                        .build();
        String input = "Large language model is";
        int maxLength = 5;
        try (ZooModel<NDList, NDList> model = criteria.loadModel();
                Predictor<NDList, NDList> predictor = model.newPredictor();
                NDManager manager = NDManager.newBaseManager("PyTorch");
                HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance("gpt2")) {

            Encoding encoding = tokenizer.encode(input);
            long[] inputIds = encoding.getIds();
            long[] attentionMask = encoding.getAttentionMask();

            NDArray use = manager.create(new boolean[] {true});
            use.setName("use_cache_branch");
            NDArray notUse = manager.create(new boolean[] {false});
            notUse.setName("use_cache_branch");
            NDList pastKeyValues = initPastKeyValues(manager, 1);

            for (int i = 0; i < maxLength; ++i) {
                NDArray useCacheBranch;
                if (i == 0) {
                    useCacheBranch = notUse;
                } else {
                    useCacheBranch = notUse;
                }
                NDArray inputArray = manager.create(inputIds).expandDims(0);
                inputArray.setName("input_ids");
                NDArray attentionMaskArray = manager.create(attentionMask).expandDims(0);
                attentionMaskArray.setName("attention_mask");

                NDList list = new NDList(inputArray, attentionMaskArray, useCacheBranch);
                list.addAll(pastKeyValues);
                NDList output = predictor.predict(list);
                // The list input here is specific to a certain model like onnx here,
                // which renders the searching algorithm work only specifically. Thus,
                // an adapter is needed here to make the searching code work for any model.

                NDArray logits = output.get(0);
                NDArray result = logits.get(new NDIndex(":,-1,:"));
                long nextToken = result.argMax().getLong();

                pastKeyValues = output.subNDList(1);
                int numLayer = pastKeyValues.size() / 2;
                for (int j = 0; j < numLayer; ++j) {
                    int index = j * 2;
                    pastKeyValues.get(index).setName("past_key_values." + j + ".key");
                    pastKeyValues.get(index + 1).setName("past_key_values." + j + ".value");
                }

                inputIds = expend(inputIds, nextToken);
                attentionMask = expend(attentionMask, 1);
            }

            logger.info(tokenizer.decode(inputIds));
        }
    }

    static long[] expend(long[] array, long item) {
        long[] ret = new long[array.length + 1];
        System.arraycopy(array, 0, ret, 0, array.length);
        ret[array.length] = item;
        return ret;
    }

    static NDList initPastKeyValues(NDManager manager, int numBatch) {
        GPTConfig config = new GPTConfig();
        long kvDim = config.kvDim;
        int numAttentionHeads = config.numAttentionHeads;
        int numLayers = config.numLayers;

        NDList list = new NDList(2 * numLayers);
        for (int i = 0; i < numLayers; ++i) {
            NDArray key = manager.zeros(new Shape(numBatch, numAttentionHeads, 1, kvDim));
            key.setName("past_key_values." + i + ".key");
            NDArray value = manager.zeros(new Shape(numBatch, numAttentionHeads, 1, kvDim));
            value.setName("past_key_values." + i + ".value");
            list.add(key);
            list.add(value);
        }
        return list;
    }
}
