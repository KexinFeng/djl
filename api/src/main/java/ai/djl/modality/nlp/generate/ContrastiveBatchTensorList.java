package ai.djl.modality.nlp.generate;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;

class ContrastiveBatchTensorList extends BatchTensorList {
    // [batch, seq_past, hiddenDim]
    // The embed vector of the past seq. seq-dim-size = |past_seq|. Will grow.
    public NDArray pastHiddenStates;

    // [batch, vacabSize]. Only the last logits, used to recall candidate token.
    public NDArray logits;

    ContrastiveBatchTensorList(NDList list, long[] seqDimOrder) {
        super();
        this.seqDimOrder = seqDimOrder;
        pastOutputIds = list.get(0);
        pastAttentionMask = list.get(1);
        pastHiddenStates = list.get(2);
        logits = list.get(3);
        pastKeyValues = list.subNDList(4);
    }

    ContrastiveBatchTensorList(
            NDArray pastOutputIds,
            NDArray pastAttentionMask,
            NDArray pastHiddenStates,
            NDArray logits,
            NDList pastKeyValues,
            long[] seqDimOrder) {
        this.pastKeyValues = pastKeyValues;
        this.pastOutputIds = pastOutputIds;
        this.pastAttentionMask = pastAttentionMask;
        this.pastHiddenStates = pastHiddenStates;
        this.logits = logits;
        this.seqDimOrder = seqDimOrder;
    }

    public ContrastiveBatchTensorList() {}

    @Override
    public ContrastiveBatchTensorList fromList(NDList inputList, long[] seqDimOrder) {
        return new ContrastiveBatchTensorList(inputList, seqDimOrder);
    }

    @Override
    public NDList getList() {
        // The pastOutputIds has to be the first in the output list
        return new NDList(pastOutputIds, pastAttentionMask, pastHiddenStates, logits)
                .addAll(pastKeyValues);
    }
}
