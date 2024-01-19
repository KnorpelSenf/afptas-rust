import java.util.ArrayList;
/**
 * Represents a rounding function in order to reduce the number of
 * different item sizes
 * 
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 */
public interface RoundingFunction{

    /**
     * Initializes the rounding function 
     * 
     * @param epsilon the approximation ratio
     */
    public void initialize(double epsilon);
    
    /**
     * Reduces the number of different item sizes by rounding several
     * items into a single one
     *
     * @param xs the original instance
     * @return a list of <code>BPItem</code> corresponding to the
     * rounded instance
     */
    public ArrayList<BPItem> round(ArrayList<BPItem> xs);

    /**
     * Constructs a solution for the original instance out of a
     * solution of the rounded instance
     *
     * @param v a solution to the rounded instance
     * @return a solution to the original instance
     */
    public Vector<Pattern<BPItem>> convert(Vector<Pattern<BPItem>> v);

    /**
     * Constructs a solution for the original instance out of a
     * solution of the rounded instance
     *
     * @param v a solution to the rounded instance
     * @return a solution to the original instance
     */
    public IVector<Pattern<BPItem>> convert(IVector<Pattern<BPItem>> v);
    
    /**
     * Returns an upper bound for the additive error produced by the
     * rounding
     */
    public int getError();
}




