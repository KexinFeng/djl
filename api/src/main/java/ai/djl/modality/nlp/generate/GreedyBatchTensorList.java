package ai.djl.modality.nlp.generate;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;

class GreedyBatchTensorList extends BatchTensorList {
    // [batch, 1]
    private NDArray nextInputIds;

    // [batch, seq_past + new_seq]
    // The cache of past attentionMask. seq-dim-size == |past_seq| + |inputIds|. Will grow.
    private NDArray pastAttentionMask;

    /* Variables below are one time step behind the above state variables. Ie, they contain all the past sequence but excludes the time step that corresponds to the above input. */

    // [batch, seq_past]. seq-dim-size == |past_seq| + |inputIds|. Will grow.
    private NDArray pastOutputIds;

    // (k, v) * numLayer,
    // kv: [batch, heads, seq_past, kvfeature]
    // The cache of past sequence. seq-dim-size == |past_seq| + |inputIds|. Will grow.
    private NDList pastKeyValues;

    GreedyBatchTensorList(
            NDArray nextInputIds,
            NDArray pastOutputIds,
            NDList pastKeyValues,
            NDArray pastAttentionMask) {
        this.nextInputIds = nextInputIds;
        this.pastKeyValues = pastKeyValues;
        this.pastOutputIds = pastOutputIds;
        this.pastAttentionMask = pastAttentionMask;
    }

    public GreedyBatchTensorList() {}

    @Override
    public BatchTensorList fromList(NDList inputList, long[] seqDimOrder) {
        return new GreedyBatchTensorList();
    }

    @Override
    public NDList getList() {
        return new NDList();
    }

    /**
     * Gets the value of the nextInputIds.
     *
     * @return the value of nextInputIds
     */
    public NDArray getNextInputIds() {
        return nextInputIds;
    }

    public void setNextInputIds(NDArray nextInputIds) {
        this.nextInputIds = nextInputIds;
    }

    /**
     * Gets the value of the pastAttentionMask.
     *
     * @return the value of pastAttentionMask
     */
    @Override
    public NDArray getPastAttentionMask() {
        return pastAttentionMask;
    }

    @Override
    public void setPastAttentionMask(NDArray pastAttentionMask) {
        this.pastAttentionMask = pastAttentionMask;
    }

    /**
     * Gets the value of the pastOutputIds.
     *
     * @return the value of pastOutputIds
     */
    @Override
    public NDArray getPastOutputIds() {
        return pastOutputIds;
    }

    @Override
    public void setPastOutputIds(NDArray pastOutputIds) {
        this.pastOutputIds = pastOutputIds;
    }

    /**
     * Gets the value of the pastKeyValues.
     *
     * @return the value of pastKeyValues
     */
    @Override
    public NDList getPastKeyValues() {
        return pastKeyValues;
    }

    @Override
    public void setPastKeyValues(NDList pastKeyValues) {
        this.pastKeyValues = pastKeyValues;
    }
}
