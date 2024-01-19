import java.util.ArrayList;
import java.util.HashMap;
import java.util.Collections;
import java.util.Comparator;

/** 
 * Implements the old robust rounding presented by Klein & Jansen in
 * "A Robust AFPTAS for Online Bin Packing with Polynomial
 * Migration"
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 */

public class OldRobustRoundingFunction implements RobustRoundingFunction{

    /**
     * the desired approximation factor
     */
    public double epsilon;

    /**
     * A mapping from the rounded items to their original items
     */
    public HashMap<BPItem, ArrayList<BPItem>> convert;

    /** 
     * A mapping from the largest item of a list to its rounded version
     */
    public HashMap<BPItem, BPItem> convertInvert;

    /**
     * A mapping describing how many times a rounded item needs to be processed
     */
    public HashMap<BPItem, Integer> count;

    /**
     * The rounding described via lists
     */
    public ArrayList<ArrayList<BPItem>> groups;

    /**
     * A simple counter to have unique names for the different rounded items
     */
    public long counter;
    
    /**
     * A counter to describe whether we need to use union or creation
     */
    public int updateCounter;
    
    /**
     * A counter to describe the number of union / creation iterations
     * already used
     */
    public int creationUnionCounter;
    
    /**
     * A counter to describe the number of creation iterations already used
     */
    public int creationCounter;

    /**
     * A counter to describe the number of union iterations already used
     */
    public int unionCounter;
    
    /**
     * The size of the first list at the beginning of the update
     */
    public int k;
    
    /**
     * Creates a new OldRobustRoundingFunction instance
     */
    public OldRobustRoundingFunction(){
        counter = 0;
    }
    
    /**
     * Initializes the rounding function
     * 
     * @param epsilon the desired approximation factor
     */
    public void initialize(double epsilon){
        this.epsilon = epsilon;
        
        groups = new ArrayList<>();
        updateCounter = 0;
        creationUnionCounter = 0;
        creationCounter = 0;
        unionCounter = 0;
    }
    
    /**
     * Rounds the item by creating 1/epsilon^2 lists of equal size
     *
     * @param xs the original instance
     * @return the rounded instance
     */
    public ArrayList<BPItem> round(ArrayList<BPItem> xs){
        int n = xs.size();

        //sort the items in descending order
        Collections.sort(xs,new Comparator<BPItem>(){
                public int compare(BPItem i1, BPItem i2){
                    return Double.compare(i2.size,i1.size);
                }});
        ArrayList<BPItem> rounded = new ArrayList<>();
        if ( n > getThreshold()){
            int k = (int)Math.ceil(n*epsilon*epsilon);
            ArrayList<BPItem> g = new ArrayList<>();
            for (BPItem it: xs){
                g.add(it);
                if (g.size() == k){
                    groups.add(g);
                    g.clear();
                }
            }
            if (g.size() < k){
                groups.add(g);
            }
        }
        // put each item into a separate group
        else{
            for(BPItem it: xs){
                ArrayList<BPItem> g = new ArrayList<>();
                g.add(it);
                groups.add(g);
            }
        }
        convert = new HashMap<>();
        convertInvert = new HashMap<>();
        count = new HashMap<>();

        // construct the rounded up items 
        for (int i = 1; i < groups.size(); i++){
            ArrayList<BPItem> gi = groups.get(i);
            BPItem large = gi.get(0);
            double multiplicity = gi.size();
            BPItem nlarge = new BPItem("Rounded"+large.id, large.size, multiplicity);
            rounded.add(nlarge);

            // update the data structures
            convert.put(nlarge,gi);
            convertInvert.put(large,nlarge);
            count.put(nlarge,(int)multiplicity);
        }
        return rounded;
    }

    /**
     * Constructs a solution to the original instance out of a solution
     * of the rounded instance and add the largest items into single
     * bins
     * 
     * @param v a solution to the rounded instance
     * @return a solution to the original instance
     */
    public Vector<Pattern<BPItem>> convert(Vector<Pattern<BPItem>> v){
        HashMap<BPItem,Integer> tmpCount = new HashMap<>();
        for (BPItem it: count.keySet()){
            tmpCount.put(it,count.get(it));
        }
        Vector<Pattern<BPItem>> res = new Vector<>();

        for(Pattern<BPItem> p: v){
            for (int l = 0; l < v.get(p); l++){
                Pattern<BPItem> np = new Pattern<>();
                for(BPItem i: p){
                    // find the original items
                    ArrayList<BPItem> g = convert.get(i);
                    int cnt = tmpCount.get(i);
                    int bound = Math.min(cnt,p.get(i));
                    for (int j = 0; j < bound; j++){
                        BPItem ni = g.get(cnt-1);
                        np.put(ni,1);
                        cnt--;
                    }
                    tmpCount.put(i,cnt);
                }
                res.put(np,Math.min(1.0,v.get(p)-l));
            }
        }
        for(BPItem it: groups.get(0)){
            Pattern<BPItem> np = new Pattern<>();
            np.put(it,1);
            res.put(np,1.0);
        }
        return res;
    }

    /**
     * Constructs a solution to the original instance out of a solution
     * of the rounded instance and add the largest items into single
     * bins
     *
     * @param v a solution to the rounded instance
     * @return a solution to the original instance
     */
    public IVector<Pattern<BPItem>> convert(IVector<Pattern<BPItem>> v){
        HashMap<BPItem,Integer> tmpCount = new HashMap<>();
        for (BPItem it: count.keySet()){
            tmpCount.put(it,count.get(it));
        }
        IVector<Pattern<BPItem>> res = new IVector<>();

        for(Pattern<BPItem> p: v){
            for (int l = 0; l < v.get(p); l++){
                Pattern<BPItem> np = new Pattern<>();
                for(BPItem i: p){
                    // find the original items
                    ArrayList<BPItem> g = convert.get(i);
                    int cnt = tmpCount.get(i);
                    int bound = Math.min(cnt,p.get(i));

                    for (int j = 0; j < bound; j++){
                        BPItem ni = g.get(cnt-1);
                        np.put(ni,1);
                        cnt--;
                    }
                    tmpCount.put(i,cnt);
                }
                if(!np.isEmpty()){
                    res.put(np,(int)Math.min(1.0,v.get(p)-l));
                }
            }
            for(BPItem it: groups.get(0)){
                Pattern<BPItem> np = new Pattern<>();
                np.put(it,1);
                res.put(np,1);
            }
        }
        return res;
    }

    /**
     * Start the rounding only when the size of the items is above this
     * 
     * @return when to use the rounding
     */
    public int getThreshold(){
        return (int) Math.ceil((1.0/(epsilon*epsilon)+2)*(1.0/epsilon+4));
    }
    
    /**
     * The value to use by improve
     * 
     * @return how much should the solution be improved
     */
    public int improve(){
        return 2;
    }
    
    /**
     * Adds a single item to the solution and the rounding
     * and adapt the rounding to it
     *
     * @param i the item to add 
     * @param pair the previous solution
     */
    public void add(BPItem i, Pair pair){
        counter++;

        // find the list where the item belongs to
        int indg;
        for(indg = 0; indg < groups.size(); indg++){
            ArrayList<BPItem> g = groups.get(indg);
            if (g.get(g.size()-1).size <= i.size){
                break;
            }
        }
        if(indg == groups.size()){
            indg--;
        }
        
        ArrayList<BPItem> g = groups.get(indg);
        
        // find the position in the list
        int pos;
        for(pos = 0; pos < g.size(); pos++){
            if(g.get(pos).size < i.size){
                break;
            }
        }
        g.add(pos,i);

        // shift the largest items from list j to list j-1 for j in [indg]
        for (int j = indg; j > 0; j--){
            // remove the item
            g = groups.get(j);
            BPItem largestG = g.get(0);
            BPItem roundG = convertInvert.get(largestG);
            g.remove(0);
            // update the solution with the new rounded item
            BPItem newRoundG = new BPItem("Ra"+counter+g.get(0).id,g.get(0).size,g.size());
            replaceAndUpdate(roundG,newRoundG,g,pair);
            groups.get(j-1).add(largestG);
        }
    }

    /**
     * Updates the rounding if necessary to maintain the invariants
     *
     * @param pair the solution to adapt to the rounding
     */
    public void update(Pair pair){
        counter++;
        updateCounter++;
        // we need to start the union / creation iteration
        if (updateCounter > groups.get(0).size()){
            k = groups.get(1).size();
            updateCounter = -Integer.MAX_VALUE;
            creationUnionCounter = (int)Math.ceil(groups.size()/2);
            creationCounter = 1;
        }
        if(creationUnionCounter > 0){
            // are we in a creation step?
            if (creationCounter > 0){
                // create the new lists
                if (creationCounter == 1){
                    groups.add(1,new ArrayList<BPItem>());
                    groups.add(1,new ArrayList<BPItem>());
                }
                // fill the lists
                BPItem l2 = groups.get(0).remove(groups.get(0).size()-1);
                BPItem l1 = groups.get(0).remove(groups.get(0).size()-1);
                groups.get(1).add(0,l2);
                groups.get(1).add(0,l1);
                
                BPItem l = groups.get(1).remove(groups.get(1).size()-1);
                groups.get(2).add(0,l);
                
                BPItem r1 = new BPItem("Ro"+counter+l1.id,l1.size,1.0);
                BPItem r2 = new BPItem("Rt"+counter+l.id,l.size,1.0);
                    
                convert.put(r1,groups.get(1));
                convert.put(r2,groups.get(2));
                convertInvert.put(l1,r1);
                convertInvert.put(l2,r2);
                count.put(r1,1);
                count.put(r2,1);

                Pattern<BPItem> p1 = new Pattern<>();
                p1.put(r1,1);
                pair.addPattern(p1);

                Pattern<BPItem> p2 = new Pattern<>();
                p1.put(r2,1);
                pair.addPattern(p2);

                // update the rounding if a rounding existed
                if (creationCounter > 1){
                    BPItem old1 = groups.get(1).get(2);
                    BPItem old2 = groups.get(2).get(1);
                    
                    BPItem rold1 = convertInvert.get(old1);
                    BPItem rold2 = convertInvert.get(old2);
                    
                    replaceAndUpdate(rold1,r1,groups.get(1),pair);
                    replaceAndUpdate(rold2,r2,groups.get(2),pair);
                }

                creationCounter++;
                // did we finish the creation operation
                if(creationCounter == k){
                    creationCounter = 0;
                    unionCounter = 1;
                }
            }
            // are we in an union step?            
            if (unionCounter > 0){
                int act = -1;
                // find the lists which is not filled properly
                for (int i = 0; i < groups.size(); i++){
                    if (groups.get(i).size() < k){
                        act = i;
                    }
                }
                // shift the largest item out of gr into gl
                for (int i = 0; i < 2; i++){
                    ArrayList<BPItem> gr = groups.get(act-(2*i));
                    ArrayList<BPItem> gl = groups.get(act-(2*i)-1);
                    
                    BPItem largestr = gr.remove(0);
                    gl.add(largestr);
                    
                    // update the rounding 
                    BPItem nlargestr = null;
                    BPItem olargestr = convertInvert.get(largestr);
                    if(! gr.isEmpty()){
                        nlargestr = new BPItem("Ru"+counter+largestr,gr.get(0).size,gr.size());
                    }
                    replaceAndUpdate(olargestr,nlargestr,gr,pair);
                }
                
                // remove all empty lists
                ArrayList<ArrayList<BPItem>> ngroups = new ArrayList<>();
                for (ArrayList<BPItem> g : groups){
                    if (!g.isEmpty()){
                        ngroups.add(g);
                    }
                }
                groups = ngroups;
                
                unionCounter++;
                // are we finished
                if (unionCounter == k){
                    unionCounter = 0;
                    creationCounter = 1;
                }
            }
            creationUnionCounter--;
            // have we finished the complete iteration
            if (creationUnionCounter == 0){
                updateCounter = 0;
                creationCounter = 0;
            }
        }
    }

    /**
     * Replaces an item by another one and updates the data structures accordingly
     *
     * @param oldI the item which is to be replaced
     * @param newI the item which replaces oldI or null if oldI should be removed only
     * @param g the rounding group corresponding to newI
     * @param pair the current solution
     */
    public void replaceAndUpdate(BPItem oldI, BPItem newI, ArrayList<BPItem> g, Pair pair){
        pair.replace(oldI,newI);

        convert.remove(oldI);
        convertInvert.remove(oldI);
        count.remove(oldI);

        if(newI != null){
            convertInvert.put(g.get(0),newI);
            convert.put(newI,g);        
            count.put(newI,g.size());
        }
    }

    /**
     * Returns an upper bound for the additive error produced by the
     * rounding
     */
    public int getError(){
        return 100*(int) (1.0/(epsilon*epsilon));
    };

}









