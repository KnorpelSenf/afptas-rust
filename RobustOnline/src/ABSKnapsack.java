import java.util.ArrayList;
import java.util.HashMap;

/**
 * This is an approximative block solver ({@link ABS}) to solve the
 * knapsack problem ({@link Knapsack}). The problem arises as a sub
 * problem by solving the bin packing problem ({@link BinPacking}).  It
 * takes a list of {@link BPItem} and returns a {@link Vector}
 * containing the next {@link Pattern} which will be used.
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 */
public class ABSKnapsack implements ABS<ArrayList<BPItem>,Pattern<BPItem>>{

    /**
     * Solves the knapsack problem with given profits and sizes to a precision of (1-t).
     * 
     * @param price the profit of the items
     * @param t the precision with which to solve the problem
     * @param input the items taken from the bin packing problem,
     * describing the sizes of the items
     * @return a {@link Vector} containing the computed, approximate
     * solution of the knapsack problem
     * 
     */
    public Vector<Pattern<BPItem>> solve(ArrayList<Double> price, double t, ArrayList<BPItem> input){

        // map the BPItems to the KPItems
        HashMap<KPItem,BPItem> ass = new HashMap<KPItem,BPItem>();
        ArrayList<KPItem> items = new ArrayList<>();
        
        // generate the knapsack instance
        for (int i = 0; i < input.size(); i++){
            BPItem item = input.get(i);
            items.add(new KPItem(item.id,price.get(i)/item.multiplicity,item.size));
            ass.put(items.get(i),item);
        }
        
        Knapsack k = new Knapsack();
        Pattern<KPItem> p = k.lawler(items,t,1);
        Pattern<BPItem> pres = new Pattern<>();
        
        // compute the solution 
        for (KPItem i: p.items.keySet()){
            pres.put(ass.get(i),p.get(i));
        }
        Vector<Pattern<BPItem>> res = new Vector<>();
        

        res.put(pres,1);
        return res;
        
    }
    
}






