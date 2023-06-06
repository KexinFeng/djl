package ai.djl.modality.nlp.generate;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;

class ContrastiveBatchTensorList extends BatchTensorList {
    // [batch, seq_past, hiddenDim]
    // The embed vector of the past seq. seq-dim-size = |past_seq|. Will grow.
    private NDArray pastHiddenStates;

    // [batch, vacabSize]. Only the last logits, used to recall candidate token.
    private NDArray logits;

    ContrastiveBatchTensorList(NDList list, long[] seqDimOrder) {
        super(list.get(0), list.get(1), list.subNDList(4), seqDimOrder);
        pastHiddenStates = list.get(2);
        logits = list.get(3);
    }

    ContrastiveBatchTensorList(
            NDArray pastOutputIds,
            NDArray pastAttentionMask,
            NDArray pastHiddenStates,
            NDArray logits,
            NDList pastKeyValues,
            long[] seqDimOrder) {
        super(pastOutputIds, pastAttentionMask, pastKeyValues, seqDimOrder);
        this.pastHiddenStates = pastHiddenStates;
        this.logits = logits;
    }

    public ContrastiveBatchTensorList() {}

    @Override
    public ContrastiveBatchTensorList fromList(NDList inputList, long[] seqDimOrder) {
        return new ContrastiveBatchTensorList(inputList, seqDimOrder);
    }

    @Override
    public NDList getList() {
        // The pastOutputIds has to be the first in the output list
        return new NDList(
                        getPastOutputIds(),
                        getPastAttentionMask(),
                        getPastHiddenStates(),
                        getLogits())
                .addAll(getPastKeyValues());
    }

    /**
     * Gets the value of the pastHiddenStates.
     *
     * @return the value of pastHiddenStates
     */
    public NDArray getPastHiddenStates() {
        return pastHiddenStates;
    }

    public void setPastHiddenStates(NDArray pastHiddenStates) {
        this.pastHiddenStates = pastHiddenStates;
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
}
