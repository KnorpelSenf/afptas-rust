import java.util.ArrayList;
/**
 * Represents a rounding function in order to reduce the number of
 * different item sizes which is suited for robust online algorithms
 * 
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 */
public interface RobustRoundingFunction extends RoundingFunction{

    /**
     * A threshold which indicates the minimal size when the
     * the rounding function becomes feasible
     *
     * @return the minimal size when to use the function
     */
    public int getThreshold();

    /**
     * Adds a new item to the rounding and the solution 
     *
     * @param i the new item to be added
     * @param p the current solution
     */
    public void add(BPItem i, Pair p);
    
    /**
     * Updates the solution and the rounding if necessary
     *
     * @param p the current solution
     */
    public void update(Pair p);
    
    /**
     * Returns the value used by a call of improve
     */
    public int improve();
    
}
