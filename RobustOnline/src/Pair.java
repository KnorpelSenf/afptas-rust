/**
 * This represents a pair of solutions for the bin packing problem. One
 * of the solutions is integral while the other one is only fractional.
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 */

public class Pair{

    /**
     * The fractional solution
     */
    Vector<Pattern<BPItem>> fractional;    

    /**
     * The integral solution
     */
    IVector<Pattern<BPItem>> integral;
    

    /**
     * Creates a new Pair consisting of two solutions
     * 
     * @param fractional the fractional solution to be stored
     * @param integral the integral solution to be stored
     */
    public Pair(Vector<Pattern<BPItem>> fractional, IVector<Pattern<BPItem>> integral){
        this.fractional = fractional;
        this.integral = integral;
    }

    /**
     * Adds a new Pattern to the solutions
     *
     * @param p the pattern to be added
     */
    public void addPattern(Pattern<BPItem> p){
        Integer tmpi = integral.get(p);
        // check whether the pattern is already present
        int tmp = 0;
        if (tmpi != null){
            tmp = tmpi;
        }
        integral.put(p,1+tmp);

        // check whether the pattern is already present
        Double di = fractional.get(p);
        double d = 0.0;
        if (di != null){
            d = di;
        }
        fractional.put(p,1+d);
    }

    /**
     * Replaces all occurrences of an item by another one
     * 
     * @param pair the solution in which to replace the item
     * @param oldItem the item which is to be replaced
     * @param newItem the item which replaces oldItem
     */
    public void replace(BPItem oldItem, BPItem newItem){

        // construct the fractional vector
        Vector<Pattern<BPItem>> fnew = new Vector<>();

        for (Pattern<BPItem> p: fractional){
            if (p.get(oldItem) != null){
                Pattern<BPItem> pn = new Pattern<>();
                for (BPItem it: p){
                    if (!it.equals(oldItem)){
                        pn.put(it,p.get(it));
                    }
                    else{
                        if(newItem != null){
                            pn.put(newItem,p.get(oldItem));
                        }
                    }
                }
                if (!pn.isEmpty()){
                    fnew.put(pn,fractional.get(p));
                }
            }
            else{
                fnew.put(p,fractional.get(p));
            }
        }

        // construct the integral vector
        IVector<Pattern<BPItem>> inew = new IVector<>();            
        for (Pattern<BPItem> p: integral){
            if (p.get(oldItem) != null){
                Pattern<BPItem> pn = new Pattern<>();
                for (BPItem it: p){
                    if (!it.equals(oldItem)){
                        pn.put(it,p.get(it));
                    }
                    else{
                        if(newItem != null){
                            pn.put(newItem,p.get(oldItem));
                        }
                    }
                }

                if (!pn.isEmpty()){
                    inew.put(pn,integral.get(p));
                }
            }
            else{
                inew.put(p,integral.get(p));
            }
        }

        integral = inew;
        fractional = fnew;
    }



}
