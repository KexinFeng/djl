package ai.djl.modality.nlp.generate;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;

class BeamBatchTensorList extends BatchTensorList {
    // [batch, beam, seq=1]
    public NDArray nextInputIds;

    // [batch, beam]
    public NDArray lastProbs;

    // [batch, beam, seq_past + new_seq]
    // The cache of past attentionMask. seq-dim-size == |past_seq| + |inputIds|. Will grow.
    public NDArray pastAttentionMask;

    /* Variables below are one time step behind the above state variables. Ie, they contain all the past sequence but excludes the time step that corresponds to the above input. */

    // [batch, beam, seq_past]. seq-dim-size == |past_seq| + |inputIds|. Will grow.
    public NDArray pastOutputIds;

    // (k, v) * numLayer,
    // kv: [batch, beam, heads, seq_past, kvfeature]
    // The cache of past sequence. seq-dim-size == |past_seq| + |inputIds|. Will grow.
    public NDList pastKeyValues;

    BeamBatchTensorList() {}

    BeamBatchTensorList(
            NDArray nextInputIds,
            NDArray pastOutputIds,
            NDList pastKeyValues,
            NDArray pastAttentionMask,
            NDArray lastProb) {
        this.nextInputIds = nextInputIds;
        this.pastKeyValues = pastKeyValues;
        this.pastOutputIds = pastOutputIds;
        this.pastAttentionMask = pastAttentionMask;
        this.lastProbs = lastProb;
    }

    @Override
    public BatchTensorList fromList(NDList inputList, long[] seqDimOrder) {
        return new BeamBatchTensorList();
    }

    @Override
    public NDList getList() {
        return new NDList();
    }
}
