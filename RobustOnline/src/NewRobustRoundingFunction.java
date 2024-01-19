import java.util.ArrayList;
import java.util.HashMap;
import java.util.Collections;
import java.util.Comparator;
/**
 * Implements the new robust rounding presented in the thesis
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 */

public class NewRobustRoundingFunction implements RobustRoundingFunction{

    /**
     * the desired approximation factor
     */
    public double epsilon;
    
    /**
     * the parameter describing the size of a list
     */
    public int k;
    
    /**
     * How many steps are still necessary to update the rounding
     */
    public int steps;

    /**
     * The sum of the sizes of all items
     */
    public double size;
    
    /**
     * A mapping from the rounded items to their original items
     */
    public HashMap<BPItem, ArrayList<BPItem>> convert;
    
    /** 
     * A mapping from the largest item of a list to its rounded version
     */
    public HashMap<BPItem,BPItem> convertInvert;

    /**
     * A mapping describing how many times a rounded item needs to be processed
     */
    public HashMap<BPItem, Integer> count;

    /**
     * The rounding described via lists
     */
    public ArrayList<ArrayList<ArrayList<BPItem>>> groups;
    
    /**
     * A simple counter to have unique names for the different rounded items
     */
    public long counter;
    /**
     * Creates a new NewRobustRoundingFunction instance
     * 
     */
    public NewRobustRoundingFunction(){
        counter = 0;
        steps = -1;
    }
    
    /**
     * Initializes the rounding function 
     *
     * @param epsilon the desired approximation factor
     */
    public void initialize(double epsilon){
        this.epsilon = epsilon;
       
        // initializes log(1/epsilon)+1 empty lists
        groups = new ArrayList<>();
        
        for(int i = 0; i <= Math.floor(log2(1.0/epsilon)); i++){
            groups.add(new ArrayList<ArrayList<BPItem>>());
        }
    }
    
    /**
     * Add an item into a list or add a new list
     * 
     * @param i the item to be added
     * @param g the groups where the item is to be inserted
     * @param ind the index of the item
     */
    public void add(BPItem i, ArrayList<ArrayList<BPItem>> g,int ind){

        int group = g.size()-1;
        if (group == -1 || g.get(group).size() == k*(int)Math.pow(2,ind)){


            ArrayList<BPItem> newList = new ArrayList<>();
            newList.add(i);
            g.add(newList);
        }
        else{
            g.get(group).add(i);
        }
    }
    
  
    /**
     * Rounds the item by putting an item with size in (2^(-t-1),2^-t]
     * in group t and splits those items in groups of size k*2^t.
     *
     * @param xs the original instance
     * @return the rounded instance
     */
    public ArrayList<BPItem> round(ArrayList<BPItem> xs){

        size = 0;
        for (BPItem i: xs){
            size = size + i.size;
        }
        // update k
        k = (int)Math.floor(size*epsilon/(2*(Math.floor(log2(2.0/epsilon))+1)));


        //sort the items in descending order
        Collections.sort(xs,new Comparator<BPItem>(){
                public int compare(BPItem i1, BPItem i2){
                    return Double.compare(i2.size,i1.size);
                }});
        
        
        // add the items to the right group
        for (BPItem i: xs){
            int ind = (int)Math.floor(-log2(i.size));
            add(i,groups.get(ind),ind);
        }

        ArrayList<BPItem> roundedInstance = new ArrayList<>();
        convert = new HashMap<>();
        convertInvert = new HashMap<>();
        count = new HashMap<>();


        // construct the rounded up items 
        for(int i = 0; i < groups.size(); i++){
            ArrayList<ArrayList<BPItem>> g = groups.get(i);
            for (int j = 0; j < g.size(); j++){

                ArrayList<BPItem> gg = g.get(j);
                double multiplicity = gg.size();
                BPItem large = gg.get(0);
                BPItem nlarge = new BPItem("Rounded"+large.id, large.size, multiplicity);
                roundedInstance.add(nlarge);
                
                // update the data structures
                convert.put(nlarge,gg);
                convertInvert.put(large,nlarge);
                count.put(nlarge,(int)multiplicity);
            }
        }
        return roundedInstance;
    }
    
    /**
     * Constructs a solution to the original instance out of a
     * solution of the rounded instance
  p   * 
     * @param v a solution to the rounded instance
     * @return a solution to the original instance
     */
    public Vector<Pattern<BPItem>> convert(Vector<Pattern<BPItem>> v){

        // copy count in order to modify it
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

                    // compute how often to replace the rounded item
                    int cnt = tmpCount.get(i);
                    int bound = Math.min(cnt,p.get(i));
                    for (int j = 0; j < bound; j++){
                        BPItem ni = g.get(cnt-1);
                        np.put(ni,1);
                        cnt--;
                    }
                    tmpCount.put(i,cnt);
                }

                // do not add empty patterns
                if(!np.isEmpty()){
                    res.put(np,Math.min(1.0,v.get(p)-l));
                }
            }
        }
        return res;
    }

    /**
     * Constructs a solution to the original instance out of a
     * solution of the rounded instance
     * 
     * @param v a solution to the rounded instance
     * @return a solution to the original instance
     */
    public IVector<Pattern<BPItem>> convert(IVector<Pattern<BPItem>> v){

        // copy count in order to modify it
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

                    // compute how often to replace the rounded item
                    int cnt = tmpCount.get(i);
                    int bound = Math.min(cnt,p.get(i));

                    for (int j = 0; j < bound; j++){
                        BPItem ni = g.get(cnt-1);
                        np.put(ni,1);
                        cnt--;
                        
                    }
                    tmpCount.put(i,cnt);
                }
                // do not add empty pattern
                if(!np.isEmpty()){
                    res.put(np,(int)Math.min(1.0,v.get(p)-l));
                }
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
        return (int) ((9*(Math.ceil(log2(1.0/epsilon))+1))/epsilon);
    }
    
    /**
     * The value to use by improve
     * 
     * @return how much should the solution be improved
     */
    public int improve(){
        return (int) Math.floor(log2(2.0/epsilon))+1;


        

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
        size = size + i.size;
        int ind = (int)Math.floor(-log2(i.size));
        ArrayList<ArrayList<BPItem>> gind = groups.get(ind);

        // find the list where the item belongs to
        int indg;
        for(indg = 0; indg < gind.size(); indg++){
            ArrayList<BPItem> g = gind.get(indg);
            if (g.get(g.size()-1).size <= i.size){
                break;
            }
        }
        if(indg == gind.size()){
            indg--;
        }
        
        ArrayList<BPItem> g = gind.get(indg);
        
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
            g = gind.get(j);
            BPItem largestG = g.get(0);
            BPItem roundG = convertInvert.get(largestG);
            g.remove(0);
            // update the solution with the new rounded item
            BPItem newRoundG = new BPItem("Ra"+counter+g.get(0).id,g.get(0).size,g.size());
            replaceAndUpdate(roundG,newRoundG,g,pair);
            gind.get(j-1).add(largestG);
        }
        
        // treat the first list specially as we do not shift an item out of it
        g = gind.get(0);
        BPItem largestFirst = g.get(0);
        if(g.size() > k*Math.pow(2,ind)){
            // we need to open a new list because the first list is full
            gind.add(0,new ArrayList<BPItem>());
            gind.get(1).remove(0);

            // we only need to update the rounding if the largest item was not the new one
            if(!largestFirst.equals(i)){
                BPItem largestOld = convertInvert.get(largestFirst);
                BPItem largestNew = new BPItem("Rl"+counter+largestOld.id,gind.get(1).get(0).size,gind.get(1).size());
                replaceAndUpdate(largestOld,largestNew,gind.get(1),pair);
            }
            // add the new rounded item 
            gind.get(0).add(largestFirst);            
            BPItem newRound = new BPItem("Rc"+counter+largestFirst.id,largestFirst.size,1.0);
            convert.put(newRound,gind.get(0));
            convertInvert.put(largestFirst,newRound);
            count.put(newRound,1);
            // add a pattern containing just the new item to the solution
            Pattern<BPItem> pa = new Pattern<>();
            pa.put(newRound,1);
            pair.addPattern(pa);
        }
        else{
            // we got a new largest item
            if(largestFirst.equals(i)){
                BPItem largestOld = convertInvert.get(g.get(1));
                BPItem largestNew = new BPItem("Rb"+counter+largestOld.id,largestFirst.size,g.size());
                replaceAndUpdate(largestOld,largestNew,g,pair);
                Pattern<BPItem> pa = new Pattern<>();
                pa.put(largestNew,1);
                pair.addPattern(pa);
            }
            else{
                // we got an additional copy of the rounded item
                BPItem nlarge = convertInvert.get(g.get(0));
                nlarge.multiplicity = nlarge.multiplicity+1;
                count.put(nlarge,(int)nlarge.multiplicity);
                // add the new item to the solution
                Pattern<BPItem> pa = new Pattern<>();
                pa.put(nlarge,1);
                pair.addPattern(pa);
            }
        }
    }
    
    
    /**
     * Updates the rounding if necessary to maintain the invariants
     *
     * @param pair the solution to adapt to the rounding
     */
    public void update(Pair pair){
        counter++;
        // we only need to do something if k changes its value
        if ((int)Math.floor(size*epsilon/(2*(Math.floor(log2(2.0/epsilon))+1))) != k){
            // find the longest list
            int max = 0;
            for (int i = 0; i < groups.size(); i++){
                ArrayList<ArrayList<BPItem>> gi = groups.get(i);
                int tmp = gi.size();
                if (tmp > max){
                    max = tmp;
                }
            }
            steps = max;
            // update k
            k = (int)Math.floor(size*epsilon/(2*(Math.floor(log2(2.0/epsilon))+1)));
        }

        if(steps >= 0){
            boolean removeLast = false;
            for(int i = 0; i < groups.size(); i++){
                ArrayList<BPItem> out = new ArrayList<>();
                ArrayList<BPItem> in = new ArrayList<>();
                ArrayList<ArrayList<BPItem>> gi = groups.get(i);
                // find the first list which has too few items
                int diff = -1;
                for(int c = 1; c < gi.size(); c++){
                    if(gi.get(c).size() < k*Math.pow(2,i)){
                        diff = c;
                    }
                }
                if (diff >= 0){
                    // shift 2^i items out of list j and into list j-1
                    for (int j = gi.size()-1; j > diff; j--){

                        ArrayList<BPItem> lasti = gi.get(j);
                        int t = (int)Math.min(Math.pow(2,i),lasti.size());
                
                        out = new ArrayList<>();
                
                        for (int k = 0; k < t; k++){
                            out.add(lasti.get(0));
                            lasti.remove(0);
                        }
                        // add the other items
                        lasti.addAll(in);
                        // if the list is empty we do must remove the corresponding item
                        if (lasti.isEmpty()){
                            removeLast = true;
                            BPItem largestOld = convertInvert.get(out.get(0));
                            replaceAndUpdate(largestOld,null,lasti,pair);
                        }
                        else{
                            BPItem largestOld = convertInvert.get(out.get(0));
                            BPItem largestNew = new BPItem("Ru"+counter+largestOld.id, lasti.get(0).size, lasti.size());
                            replaceAndUpdate(largestOld,largestNew,lasti,pair);
                        }
                        
                        in = out;
                    }

                    // treat the first list separately
                    ArrayList<BPItem> g = gi.get(diff);
                    g.addAll(in);
                    // update the rounding
                    BPItem largest = convertInvert.get(g.get(0));
                    largest.multiplicity = largest.multiplicity + in.size();
                    count.put(largest,(int)largest.multiplicity);
                    Pattern<BPItem> pa = new Pattern<>();
                    pa.put(largest,in.size());
                    pair.addPattern(pa);
                
                    if (removeLast){
                        gi.remove(gi.size()-1);
                    }
                }
            }
            steps--;
        }
    }

    /**
     * Computes the logarithm to the base of 2
     * 
     * @param num the argument of the logarithm
     * @return log_2(num)
     */
    public double log2(double num){
        return Math.log(num)/Math.log(2);
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
        return 100*(int) ((1.0/epsilon)*log2(1.0/epsilon));
    };
}








