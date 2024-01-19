import java.util.ArrayList;
import java.util.HashMap;


/**
 * Solves the fractional bin packing problem via {@link MaxMin} 
 * 
 * @author Sebastian Berndt <seb@informatik.uni-kiel.de>
 */
public class FBinPacking{
    
    /**
     * Creates a new FBinPacking instance
     * 
     */
    public FBinPacking(){}
    
    /**
     * Solves the fractional bin packing up to a given precision
     *
     * @param items a list of items
     * @param epsilon the precision to be reached
     * @return a {@link Vector} containing the used {@link Pattern}s
     */
    public Vector<Pattern<BPItem>> solve(ArrayList<BPItem> items, double epsilon){
        

        // create the functions given by the items
        ArrayList<Function<Pattern<BPItem>>> fs = new ArrayList<>();
        
        for (BPItem i: items){
            final BPItem item = i;
            fs.add(new Function<Pattern<BPItem>>(){
                    public double apply(Vector<Pattern<BPItem>> param){
                        double sum = 0;
                        for (Pattern<BPItem> p: param.map.keySet()){
                            if (p.items.containsKey(item)){
                                sum = sum+p.get(item)*param.get(p);
                            }
                        }
                        sum = sum/item.multiplicity;
                        return sum;
                    }});
        }
        ABS<ArrayList<BPItem>,Pattern<BPItem>> abs = new ABSKnapsack();
        MaxMin<ArrayList<BPItem>,Pattern<BPItem>> max = new MaxMin<>(abs,items);


        // solve the problem
        Vector<Pattern<BPItem>> res = max.maxmin(fs,epsilon);
        
        // compute the minimum of the function
        double min = Double.MAX_VALUE;
        for (Function<Pattern<BPItem>> f: fs){
            double tmp = f.apply(res);

            if (tmp < min){
                min = tmp;
            }
        }
        // scale the result accordingly
        Vector<Pattern<BPItem>> sres = res.scale((1/min));

        ReduceComponents red = new ReduceComponents();
        Vector<Pattern<BPItem>> reducedRes = red.reduce(sres,items);
        

        String wit = System.getenv("WITNESS");
        if (wit != null && wit.equals("true")){
            FBinPackingWitness fbpw = new FBinPackingWitness();
            if (!fbpw.check(items,reducedRes)){
                System.out.println("Could not witness");
                System.exit(1);
            }
        }

        return sres;
        
    }

    
}












