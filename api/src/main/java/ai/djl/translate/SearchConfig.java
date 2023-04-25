package ai.djl.translate;

public class SearchConfig {

    public int k;
    public float alpha;
    public int beam;
    public int maxSeqLength;
    public long padTokenId;
    public long eosTokenId;
    public boolean suffixPadding;

    /** Constructs a new ContrastiveSearchConfig object with default values. */
    public SearchConfig() {
        this.k = 4;
        this.alpha = 0.6f;
        this.beam = 3;
        this.maxSeqLength = 30;
        this.eosTokenId = 50256;
        this.padTokenId = 50256;
        this.suffixPadding = true;
    }
}