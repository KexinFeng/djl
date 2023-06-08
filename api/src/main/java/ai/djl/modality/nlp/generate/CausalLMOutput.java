package ai.djl.modality.nlp.generate;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;

/** CausalLMOuput is used to contain multiple output of a language model. */
public class CausalLMOutput {

    // [batch, seq, feature]
    // The prob. conditional on a sequence that ends at an element in seq-dim. seq-dim-size =
    // |inputIds|
    private NDArray logits;

    // [batch, seq, dim] * (layers+1) -> take -1
    // The vec. rep. of a sequence that ends at an element in seq-dim. seq-dim-size = |inputIds|
    private NDArray hiddenStates;

    // (k, v) * numLayer,
    // kv: [batch, heads, seq_past, feature]
    // The cache of past sequence. seq-dim-size == |seq_past| + |inputIds|
    private NDList pastKeyValuesList;

    public CausalLMOutput(NDArray logits, NDList pastKeyValues) {
        this.logits = logits;
        this.pastKeyValuesList = pastKeyValues;
    }

    public CausalLMOutput(NDArray logits, NDArray hiddenState, NDList pastKeyValueList) {
        this.logits = logits;
        this.pastKeyValuesList = pastKeyValueList;
        this.hiddenStates = hiddenState;
    }

    /**
     * Gets the value of the logits.
     *
     * @return the value of logits
     */
    public NDArray getLogits() {
        return logits;
    }

    public void setLogits(NDArray logits) {
        this.logits = logits;
    }

    /**
     * Gets the value of the allHiddenStates.
     *
     * @return the value of allHiddenStates
     */
    public NDArray getHiddenState() {
        return hiddenStates;
    }

    /**
     * Gets the value of the pastKeyValuesList.
     *
     * @return the value of pastKeyValuesList
     */
    public NDList getPastKeyValuesList() {
        return pastKeyValuesList;
    }
}
