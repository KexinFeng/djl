package ai.djl.modality.nlp.generate;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;

// BatchTensorList represents a search state, and the NDArrays inside are updated in each iteration
// of the
// autoregressive loop.
// It is a struct consisting of NDArrays, whose first dimension is batch, and also contains
// sequence dimension (whose position in tensor's shape is specified by seqDimOrder).
// The SeqBatcher batch operations will operate on these two dimensions.
public abstract class BatchTensorList {
    // [batch, seq_past]. seq-dim-size == |past_seq| + |inputIds|. Will grow.
    public NDArray pastOutputIds;

    // [batch, seq_past]
    // The cache of past attentionMask. seq-dim-size == |past_seq| + |inputIds|. Will grow.
    public NDArray pastAttentionMask;

    // (k, v) * numLayer,
    // kv: [batch, heads, seq_past, kvfeature]
    // The cache of past sequence. seq-dim-size == |past_seq| + |inputIds|. Will grow.
    public NDList pastKeyValues;

    // Sequence dimension order among all dimensions for each element in the batch list.
    public long[] seqDimOrder;

    BatchTensorList() {}

    BatchTensorList(NDList list, long[] seqDimOrder) {
        this.seqDimOrder = seqDimOrder;
        pastOutputIds = list.get(0);
        pastAttentionMask = list.get(1);
        pastKeyValues = list.subNDList(2);
    }

    BatchTensorList(
            NDArray pastOutputIds,
            NDArray pastAttentionMask,
            NDList pastKeyValues,
            long[] seqDimOrder) {
        this.pastKeyValues = pastKeyValues;
        this.pastOutputIds = pastOutputIds;
        this.pastAttentionMask = pastAttentionMask;
        this.seqDimOrder = seqDimOrder;
    }

    public abstract BatchTensorList fromList(NDList inputList, long[] seqDimOrder);

    // The pastOutputIds has to be the first in the output list
    public abstract NDList getList();

    public long[] getSeqDimOrder() {
        return seqDimOrder;
    }
}
