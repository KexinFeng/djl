package ai.djl.modality.nlp.generate;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

// This is a scheduler, serving as an API to the consumer of the system, allowing for three major
// actions: initForward, addBatch, fastForward, collectResults.
// An optimal control sequence should be solved, after considering the time consumption of each
// action, the batch size and sequence length of queueing requests. Such optimal control solver
// needs additional effort. Primitive policy is setting several thresholds.
public abstract class SeqBatchScheduler {
    private static final Logger logger = LoggerFactory.getLogger(SeqBatchScheduler.class);

    LMBlock lmBlock;
    SeqBatcher seqBatcher;

    NDManager manager;

    SearchConfig config;

    Map<Long, NDArray> results;

    public SeqBatchScheduler(LMBlock lmBlock, SearchConfig config) {
        this.lmBlock = lmBlock;
        this.config = config;
        results = new ConcurrentHashMap<>();
    }

    /**
     * Initialize the iteration and SeqBatcher
     *
     * @return SeqBatcher. Stores the search state and operate on the BatchTensorList.
     */
    public abstract SeqBatcher initForward(NDArray inputIds, NDArray batchUids);

    /**
     * Go forward for a given number of iterations.
     *
     * @return boolean. Indicate whether the Batch is empty.
     */
    public boolean incrementForward(int count) {
        int i = 0;
        while (i++ < count) {
            if (seqBatcher == null || seqBatcher.getData() == null) {
                logger.info(
                        "seqBatcher not set or is empty. Please call addBatch. Current inference"
                                + " times is "
                                + i);
                return true;
            }

            inferenceCall();
            if (seqBatcher.sequenceComplete()) {
                results.putAll(seqBatcher.collectAndTrim());
            }
        }
        return false;
    }

    abstract NDArray inferenceCall();

    /** Add new batch. */
    public void addRequest(NDArray inputIds, NDArray batchUids) {
        SeqBatcher seqBatcherNew = initForward(inputIds, batchUids);
        if (seqBatcher == null) {
            seqBatcher = seqBatcherNew;
        } else {
            seqBatcher.addBatch(seqBatcherNew);
        }
    }

    /** Collect finished results. */
    public Map<Long, NDArray> collectResults() {
        Map<Long, NDArray> output = results;
        results = new ConcurrentHashMap<>();
        return output;
    }

    static NDArray computeOffSets(NDArray inputIds, SearchConfig config) {
        int numBatch = (int) inputIds.getShape().get(0);
        int initSeqSize = (int) inputIds.getShape().get(1);

        // Linear search from left to find the first position that's not padTokenId.
        long[] offSetsArray = new long[numBatch];
        for (int i = 0; i < numBatch; i++) {
            long[] aSequence = inputIds.get("{},:", i).toLongArray();
            int idx = 0;
            while (idx < initSeqSize) {
                if (aSequence[idx] != config.padTokenId) {
                    break;
                }
                idx++;
            }
            offSetsArray[i] = idx;
        }

        NDManager manager = inputIds.getManager();
        return manager.create(offSetsArray).reshape(-1, 1);
    }

    static NDArray computeAttentionMask(NDArray inputIds, SearchConfig config) {
        int numBatch = (int) inputIds.getShape().get(0);
        int initSeqSize = (int) inputIds.getShape().get(1);

        NDManager manager = inputIds.getManager();
        NDArray attentionMask =
                manager.ones(new Shape(1, inputIds.getShape().get(-1)), DataType.INT64)
                        .reshape(1, -1)
                        .repeat(0, numBatch);

        // Linear search to find the offset and set the mask
        for (int i = 0; i < numBatch; i++) {
            long[] aSequence = inputIds.get("{},:", i).toLongArray();
            int idx = 0;
            while (idx < initSeqSize) {
                if (aSequence[idx] != config.padTokenId) {
                    break;
                }
                idx++;
            }
            attentionMask.set(new NDIndex("{},{}:{}", i, 0, idx), 0);
        }

        // [batch, pastSeq]
        return attentionMask;
    }

    static NDArray computePositionIds(
            NDArray inputIds, NDArray offSets, long pastSeqLength, int repeat) {
        NDManager manager = inputIds.getManager();
        NDArray positionIds =
                manager.arange(
                                pastSeqLength,
                                pastSeqLength + inputIds.getShape().get(-1),
                                1,
                                DataType.INT64)
                        .expandDims(0)
                        .repeat(0, inputIds.getShape().get(0));

        NDArray positionIdsShifted = positionIds.subi(offSets.reshape(-1, 1).repeat(0, repeat));
        positionIds = positionIdsShifted.maximum(positionIdsShifted.zerosLike());

        // [batch, inputSeq]
        return positionIds;
    }
}
