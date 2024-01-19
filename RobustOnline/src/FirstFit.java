import java.util.HashMap;
import java.util.ArrayList;

/**
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 * Solves the Bin Packing problem via the simple first fit heuristic
 */

public class FirstFit{
    
    /**
     * A mapping between the pattern filled with small items
     * and the corresponding patterns without small items
     */
    HashMap<Pattern<BPItem>,Pattern<BPItem>> belongsToInvers;
    
    /**
     * Creates a new FirstFit instance
     */
    public FirstFit(){}

    /**
     * Adds some items to an existing solution by simply using the first
     * free place
     *
     * @param sol the existing solution
     * @param small the items to be added
     * @return a modified solution which packs all items
     */
    public IVector<Pattern<BPItem>> firstFit(IVector<Pattern<BPItem>> sol, ArrayList<BPItem> small){
        belongsToInvers = new HashMap<>();
        HashMap<Pattern<BPItem>,Double> free = freeSpaces(sol);
        for(BPItem i: small){
            boolean found = false;
            Pattern<BPItem> place = new Pattern<>();
            // search for a pattern which has enough space
            for(Pattern<BPItem> p: sol){
                if (free.get(p) >= i.size){
                    place = p;
                    found = true;
                }
            }
            // we found a fitting space
            if (found){
                sol.put(place,sol.get(place)-1);
                double freeSpace = free.get(place);
                place.put(i,1);
                sol.put(place,1);
                free.put(place,freeSpace-i.size);
            }
            // we did not found a fitting space and need to open a new
            // bin
            else{
                Pattern<BPItem> p = new Pattern<>();
                p.put(i,1);
                sol.put(p,1);
                free.put(p,1-i.size);
            }
        }
        for(Pattern<BPItem> p: sol){
            Pattern<BPItem> pw = new Pattern<>(p);
            for (BPItem i: small){
                pw.put(i,0);
            }
            Pattern<BPItem> ptmp = p;
            Pattern<BPItem> ptmp2 = p;
            // update belongsToInvers in order to reduce its size
            while(belongsToInvers.get(ptmp) != null){
                ptmp2 = ptmp;
                ptmp = belongsToInvers.get(ptmp);
                belongsToInvers.remove(ptmp2);

            }
            belongsToInvers.put(pw,ptmp);
        }
        return sol;
    }

    /**
     * Compute the free spaces of a vector of patterns
     *
     * @param v the {@link Vector} containing the {@link Pattern}
     * @return a {@link HashMap} which stores the available space in each pattern
     */
    public HashMap<Pattern<BPItem>,Double> freeSpaces(IVector<Pattern<BPItem>> v){
        HashMap<Pattern<BPItem>,Double> h = new HashMap<>();
        for(Pattern<BPItem> p: v){

            double space = 0;
            for (BPItem i: p){

                space = space + (i.size*p.get(i));
            }
            h.put(p,1-space);
        }
        return h;
    }
}
