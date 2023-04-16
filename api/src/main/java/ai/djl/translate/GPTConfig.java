package ai.djl.translate;

/** GPTConfig is used to store the GPT parameters used to select different versions of GPT. */
public class GPTConfig {
    public int numAttentionHeads;
    public int numLayers;
    public long hiddenStateDim;
    public long logitsDim;
    public long kvDim;

    public GPTConfig() {
        kvDim = 64;
        logitsDim = 50257;
        numAttentionHeads = 12;
        numLayers = 12;
        hiddenStateDim = 768;
    }
}
