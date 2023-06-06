package ai.djl.modality.nlp.generate;

/** GPTConfig is used to store the GPT parameters used to select different versions of GPT. */
public class GPTConfig {
    private int numAttentionHeads;
    private int numLayers;
    private long kvDim;

    public GPTConfig() {
        numAttentionHeads = 12;
        numLayers = 12;
        kvDim = 64;
    }

    /**
     * Gets the value of the numAttentionHeads.
     *
     * @return the value of numAttentionHeads
     */
    public int getNumAttentionHeads() {
        return numAttentionHeads;
    }

    /**
     * Gets the value of the numLayers.
     *
     * @return the value of numLayers
     */
    public int getNumLayers() {
        return numLayers;
    }

    public void setNumLayers(int numLayers) {
        this.numLayers = numLayers;
    }

    /**
     * Gets the value of the kvDim.
     *
     * @return the value of kvDim
     */
    public long getKvDim() {
        return kvDim;
    }
}
