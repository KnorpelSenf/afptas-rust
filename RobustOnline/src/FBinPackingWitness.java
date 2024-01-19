import java.util.ArrayList;
import java.util.HashMap;
/**
 * Checks whether a fractional bin packing solution is feasible
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 */

public class FBinPackingWitness{

    /**
     * Creates a new FBinPackingWitness instance
     */
    public FBinPackingWitness(){
    }
    
    /**
     * Checks whether a given vector is a feasible solution of the
     * fractional bin packing problem
     *
     * @param items a list of items
     * @param res a possible solution of the problem
     * @return <code>true</code> if all items are packed totally in the solution;
     *         <code>false</code> otherwise
     */
    public boolean check(ArrayList<BPItem> items, Vector<Pattern<BPItem>> res){

        // sum up the occurrences of the items
        HashMap<BPItem,Double> f = new HashMap<>();
        for (Pattern<BPItem> p: res){
            for (BPItem i: p){
                if (f.containsKey(i)){
                    f.put(i,f.get(i)+res.get(p)*p.get(i));
                }
                else{
                    f.put(i,res.get(p)*p.get(i));
                }
            }
        }
        
        boolean b = true;
        for (BPItem i: items){
            // we are dealing with floats!
            if (f.get(i) < 0.9999){
                b = false;
            }
        }
        
        return b;
    }
}
