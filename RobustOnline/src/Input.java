/**
 * An interface to describe how to get the next item for an online bin
 * packing instance
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 */
public interface Input{

    /**
     * Return the next item
     *
     * @return the next item
     */
    public BPItem getNextItem();



}
