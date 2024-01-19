import java.util.ArrayList;
/**
 * Checks whether a bin packing solution is good enough
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 */

public class BinPackingWitness{

    /**
     * An upper bound for the problem
     */
    int value;

    /**
     * Creates a new BinPackingWitness instance
     * 
     * @param xs the bin packing instance
     */
    public BinPackingWitness(ArrayList<BPItem> xs){
        // compute the simple lower bound
        ArrayList<ArrayList<BPItem>> g = new ArrayList<>();
        for (BPItem it: xs){
            for (int i = 0; i < g.size(); i++){
                if (freeSpace(g.get(i)) >= it.size){
                    g.get(i).add(it);
                    break;
                }
            }
        }
        value = g.size();
    }

    /**
     * Computes the free space left in a bin
     *
     * @param g the bin 
     * @return the free space left in the bin
     */
    public double freeSpace(ArrayList<BPItem> g){
        double space = 1.0;
        for (BPItem it: g){
            space = space - it.size;
        }
        return space;
    }

    /**
     * Tests whether a solution to the bin packing problem is within a
     * given intervall and feasible
     * 
     * @param v the solution of the problem
     * @param epsilon the desired approximation factor
     * @param C the additive error which is allowed
     * @return whether the solution is feasible and good enough
     */
    public boolean check(IVector<Pattern<BPItem>> v, double epsilon, int C){
        int sol = (int) v.sum();
        return (((1+epsilon)/2.0 * value <= sol) && (sol <= (1+epsilon)*value+C));
    }


}
