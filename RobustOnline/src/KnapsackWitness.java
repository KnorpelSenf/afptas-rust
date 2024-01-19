import java.util.ArrayList;
/**
 * Checks whether a knapsack solution is good enough
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 */

public class KnapsackWitness{

    /**
     * The lower bound of the problem
     */
    double p0;
    
    /**
     * Creates a new KnapsackWitness instance
     *
     * @param xs the items of the problem
     * @param b the capacity of the knapsack
     */
    public KnapsackWitness(ArrayList<KPItem> xs, int b){
        // compute the lower bound
        double pmax = 0;
        double ratio = -1;
        KPItem ratioMax = null;
        for (KPItem it: xs){
            if (it.profit > pmax){
                pmax = it.profit;
            }
            if (it.profit/it.size > ratio){
                ratio = it.profit/it.size;
                ratioMax = it;
            }
        }
        p0 = Math.max(pmax,ratioMax.profit*(Math.floor(b/ratioMax.size)));
    }
    
    /**
     * Tests whether a solution to the knapsack problem is within a
     * given intervall and feasible
     *
     * @param sol the solution of the problem
     * @param epsilon the desired approximation factor
     * @param b the capacity of the knapsack 
     * @return whether the solution is feasible and good enough
     */
    public boolean check(Pattern<KPItem> sol, double epsilon, int b){
        double size = 0;
        double value = 0;
        for (KPItem it: sol){
            value = value+sol.get(it)*it.profit;
            size = size+sol.get(it)*it.size;
        }
        return (((1-epsilon)*p0 <= value) && (value <= 2*(1-epsilon)*p0) && (size <= b));
    }
}
